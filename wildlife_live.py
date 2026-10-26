#!/usr/bin/env python3
import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["elephant", "wild_boar", "leopard", "bear"]
MODEL_DEFAULT = "~/wildlife4_deploy/wildlife4_v4_demo.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Live four-class wildlife detection using Raspberry Pi CSI camera"
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(os.path.expanduser(args.model))
    if not model_path.is_file():
        raise SystemExit(f"ERROR: Model not found: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("ERROR: --conf must be between 0 and 1")

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"Model: {model_path}", flush=True)
    print(f"Provider: {session.get_providers()}", flush=True)
    print(f"Confidence threshold: {args.conf:.2f}", flush=True)
    print("Starting CSI camera. Press Ctrl+C to stop.", flush=True)

    command = [
        "rpicam-vid",
        "--nopreview",
        "--timeout", "0",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.fps),
        "--codec", "mjpeg",
        "--output", "-",
        "--flush",
    ]

    stop_reader = threading.Event()
    lock = threading.Lock()
    latest = {"frame": None, "sequence": 0}

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    def read_camera_stream():
        buffer = bytearray()
        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"

        try:
            while not stop_reader.is_set():
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)

                while True:
                    start = buffer.find(soi)
                    if start < 0:
                        # Retain a possible first byte of a split JPEG marker.
                        buffer[:] = b"\xff" if buffer.endswith(b"\xff") else b""
                        break

                    if start > 0:
                        del buffer[:start]

                    end = buffer.find(eoi, 2)
                    if end < 0:
                        break

                    jpeg = bytes(buffer[:end + 2])
                    del buffer[:end + 2]
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        with lock:
                            latest["frame"] = frame
                            latest["sequence"] += 1
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=read_camera_stream, daemon=True)
    reader.start()

    last_sequence = 0
    last_report = time.monotonic()
    processed_since_report = 0

    try:
        while True:
            with lock:
                sequence = latest["sequence"]
                frame = latest["frame"]

            if sequence == last_sequence or frame is None:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Camera process exited with code {process.returncode}"
                    )
                time.sleep(0.005)
                continue

            # Only process the newest available frame.
            last_sequence = sequence
            h, w = frame.shape[:2]

            scale = min(640.0 / w, 640.0 / h)
            nw = round(w * scale)
            nh = round(h * scale)
            resized = cv2.resize(frame, (nw, nh))

            left = (640 - nw) // 2
            top = (640 - nh) // 2
            canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
            canvas[top:top + nh, left:left + nw] = resized

            tensor = cv2.cvtColor(
                canvas, cv2.COLOR_BGR2RGB
            ).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: tensor})
            inference_ms = (time.perf_counter() - start_time) * 1000.0

            pred = np.asarray(outputs[0])
            if pred.ndim == 3:
                pred = pred[0]
            if pred.ndim != 2 or pred.shape[1] != 6:
                raise RuntimeError(
                    f"Unexpected model output shape: {outputs[0].shape}"
                )

            detections = []
            for row in pred:
                if not np.isfinite(row).all():
                    continue

                x1, y1, x2, y2, score, class_value = map(float, row)
                if score < args.conf:
                    continue

                class_id = int(class_value)
                if class_id < 0 or class_id >= len(CLASSES):
                    continue

                # Undo letterboxing and clamp boxes to the source frame.
                x1 = max(0, min(w, round((x1 - left) / scale)))
                x2 = max(0, min(w, round((x2 - left) / scale)))
                y1 = max(0, min(h, round((y1 - top) / scale)))
                y2 = max(0, min(h, round((y2 - top) / scale)))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    f"{CLASSES[class_id]} {score:.2f} "
                    f"[{x1},{y1},{x2},{y2}]"
                )

            processed_since_report += 1
            now = time.monotonic()
            if now - last_report >= 1.0:
                rate = processed_since_report / (now - last_report)
                description = "; ".join(detections) if detections else "no detections"
                print(
                    f"camera_frames={sequence} | "
                    f"inference_rate={rate:.1f} FPS | "
                    f"inference={inference_ms:.0f} ms | {description}",
                    flush=True,
                )
                last_report = now
                processed_since_report = 0

    except KeyboardInterrupt:
        print("\nStopping wildlife detector...", flush=True)
    finally:
        stop_reader.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()
        reader.join(timeout=2)


if __name__ == "__main__":
    main()
