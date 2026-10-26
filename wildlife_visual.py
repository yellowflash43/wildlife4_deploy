#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["elephant", "wild_boar", "leopard", "bear"]
COLORS = [
    (255, 180, 0),
    (0, 220, 0),
    (0, 200, 255),
    (255, 0, 255),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="~/wildlife4_deploy/wildlife4_v4_demo.onnx",
    )
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    model = Path(os.path.expanduser(args.model))
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    if not 0 <= args.conf <= 1:
        raise SystemExit("Confidence must be between 0 and 1")

    session = ort.InferenceSession(
        str(model), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    print(f"Model: {model}", file=sys.stderr, flush=True)
    print("Annotated MJPEG stream starting.", file=sys.stderr, flush=True)

    command = [
        "rpicam-vid", "--nopreview", "--timeout", "0",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.fps),
        "--codec", "mjpeg", "--output", "-", "--flush",
    ]

    camera = subprocess.Popen(command, stdout=subprocess.PIPE)
    buffer = bytearray()
    frames = 0
    last_report = time.monotonic()
    last_ms = 0.0

    try:
        while True:
            chunk = camera.stdout.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)

            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer[:] = b"\xff" if buffer.endswith(b"\xff") else b""
                    break
                if start:
                    del buffer[:start]

                end = buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    break

                jpg = bytes(buffer[:end + 2])
                del buffer[:end + 2]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                scale = min(640 / w, 640 / h)
                nw, nh = round(w * scale), round(h * scale)
                resized = cv2.resize(frame, (nw, nh))
                left, top = (640 - nw) // 2, (640 - nh) // 2

                canvas = np.full((640, 640, 3), 114, np.uint8)
                canvas[top:top + nh, left:left + nw] = resized
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

                t0 = time.perf_counter()
                output = session.run(
                    None, {input_name: tensor}
                )[0]
                last_ms = (time.perf_counter() - t0) * 1000

                count = 0
                for row in output[0]:
                    if not np.isfinite(row).all():
                        continue

                    x1, y1, x2, y2, score, cls = map(float, row)
                    class_id = int(cls)
                    if score < args.conf or not 0 <= class_id < len(CLASSES):
                        continue

                    x1 = max(0, min(w - 1, round((x1 - left) / scale)))
                    x2 = max(0, min(w - 1, round((x2 - left) / scale)))
                    y1 = max(0, min(h - 1, round((y1 - top) / scale)))
                    y2 = max(0, min(h - 1, round((y2 - top) / scale)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    color = COLORS[class_id]
                    label = f"{CLASSES[class_id]} {score:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    ty = max(20, y1 - 7)
                    cv2.putText(
                        frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        cv2.LINE_AA,
                    )
                    count += 1

                cv2.putText(
                    frame, f"Pi wildlife | {last_ms:.0f} ms | conf {args.conf:.2f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA,
                )

                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ok:
                    sys.stdout.buffer.write(encoded.tobytes())
                    sys.stdout.buffer.flush()

                frames += 1
                now = time.monotonic()
                if now - last_report >= 5:
                    print(
                        f"Frames={frames}, last_detections={count}, "
                        f"inference={last_ms:.0f} ms",
                        file=sys.stderr, flush=True,
                    )
                    last_report = now

    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if camera.stdout:
            camera.stdout.close()
        if camera.poll() is None:
            camera.terminate()
            try:
                camera.wait(timeout=3)
            except subprocess.TimeoutExpired:
                camera.kill()
                camera.wait()


if __name__ == "__main__":
    main()
