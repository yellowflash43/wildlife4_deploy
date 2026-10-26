#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL = Path.home() / "wildlife4_deploy/wildlife4_v4_demo.onnx"
CLASSES = ["elephant", "wild_boar", "leopard", "bear"]
COLORS = [(255, 180, 0), (0, 220, 0), (0, 200, 255), (255, 0, 255)]
CONFIDENCE = 0.50
WIDTH, HEIGHT, FPS = 640, 480, 5
PORT = 8080

stop = threading.Event()
raw_lock = threading.Lock()
raw = {"seq": 0, "frame": None}
view_lock = threading.Lock()
view = {"seq": 0, "jpg": None, "detections": [], "fps": 0.0, "ms": 0.0}
server_ref = {"server": None}


def capture_camera():
    command = [
        "rpicam-vid", "--nopreview", "--timeout", "0",
        "--width", str(WIDTH), "--height", str(HEIGHT),
        "--framerate", str(FPS), "--codec", "mjpeg",
        "--output", "-", "--flush",
    ]

    while not stop.is_set():
        proc = None
        try:
            print("Starting Raspberry Pi camera...", flush=True)
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)
            buffer = bytearray()

            while not stop.is_set():
                chunk = proc.stdout.read(8192)
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

                    jpeg = bytes(buffer[:end + 2])
                    del buffer[:end + 2]
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        with raw_lock:
                            raw["seq"] += 1
                            raw["frame"] = frame

            if not stop.is_set():
                print("Camera stream ended; retrying shortly.", flush=True)
        except Exception as exc:
            print(f"Camera error: {exc}", flush=True)
        finally:
            if proc is not None:
                if proc.stdout:
                    proc.stdout.close()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

        if not stop.is_set():
            stop.wait(2)


def inference_worker(session):
    input_name = session.get_inputs()[0].name
    last_seq = 0
    last_time = None
    count = 0
    last_report = time.monotonic()

    while not stop.is_set():
        with raw_lock:
            seq = raw["seq"]
            frame = raw["frame"]

        if frame is None or seq == last_seq:
            stop.wait(0.02)
            continue

        last_seq = seq
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        left, top = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = tensor.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        ms = (time.perf_counter() - t0) * 1000
        pred = np.asarray(outputs[0])
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2 or pred.shape[1] != 6:
            print(f"Unexpected model output shape: {outputs[0].shape}", flush=True)
            continue

        detections = []
        for row in pred:
            if not np.isfinite(row).all():
                continue
            x1, y1, x2, y2, score, cls = map(float, row)
            class_id = int(cls)
            if score < CONFIDENCE or not 0 <= class_id < len(CLASSES):
                continue

            x1 = max(0, min(w - 1, round((x1 - left) / scale)))
            x2 = max(0, min(w - 1, round((x2 - left) / scale)))
            y1 = max(0, min(h - 1, round((y1 - top) / scale)))
            y2 = max(0, min(h - 1, round((y2 - top) / scale)))
            if x2 <= x1 or y2 <= y1:
                continue

            label = f"{CLASSES[class_id]} {score:.2f}"
            detections.append({"class": CLASSES[class_id], "confidence": round(score, 3)})
            color = COLORS[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            ty = max(20, y1 - 7)
            cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

        now = time.monotonic()
        if last_time is not None and now > last_time:
            instant_fps = 1.0 / (now - last_time)
            view_fps = instant_fps
        else:
            view_fps = 0.0
        last_time = now

        cv2.rectangle(frame, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(frame, f"Wildlife AI | {view_fps:.1f} FPS | {ms:.0f} ms",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with view_lock:
                view["seq"] += 1
                view["jpg"] = encoded.tobytes()
                view["detections"] = detections
                view["fps"] = round(view_fps, 1)
                view["ms"] = round(ms, 1)

        count += 1
        if now - last_report >= 5:
            names = ", ".join(d["class"] for d in detections) or "none"
            print(f"Inference running: {count} frames processed; latest detections: {names}",
                  flush=True)
            last_report = now


PAGE = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi Wildlife Vision</title>
<style>
body{margin:0;background:#101418;color:#f4f7fa;font:16px sans-serif;text-align:center}
header{padding:12px}h2{margin:4px}p{color:#c4cbd3;margin:8px}
img{display:block;width:100%;height:auto;max-height:78vh;object-fit:contain;background:#000}
small{color:#aeb8c2}
</style></head><body>
<header><h2>Pi Wildlife Vision</h2>
<p>Live camera with AI detection</p><small>Elephant · Wild boar · Leopard · Bear</small></header>
<img src="/stream.mjpg" alt="Waiting for Pi camera stream">
<p id="status">Connecting to camera...</p>
<script>
async function update(){try{
const r=await fetch('/health',{cache:'no-store'});const d=await r.json();
document.getElementById('status').textContent=
'Camera frames: '+d.frames+' | Inference: '+d.inference_ms+' ms | Detected: '+
(d.detections.length?d.detections.map(x=>x.class+' '+x.confidence).join(', '):'nothing above threshold');
}catch(e){document.getElementById('status').textContent='Connecting...'}}
setInterval(update,2000);update();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(PAGE.encode("utf-8"))
            return

        if self.path == "/health":
            with view_lock:
                with raw_lock:
                    frames = raw["seq"]
                data = {
                    "frames": frames,
                    "inference_ms": view["ms"],
                    "fps": view["fps"],
                    "detections": view["detections"],
                }
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            last = -1
            try:
                while not stop.is_set():
                    with view_lock:
                        seq, jpg = view["seq"], view["jpg"]
                    if jpg is None or seq == last:
                        time.sleep(0.03)
                        continue
                    last = seq
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n" + jpg + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return

        self.send_error(404)


def main():
    if not MODEL.is_file():
        raise SystemExit(f"Model not found: {MODEL}")

    print(f"Loading model {MODEL}", flush=True)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    print("Model loaded; starting camera and phone web server.", flush=True)

    threading.Thread(target=capture_camera, daemon=True).start()
    threading.Thread(target=inference_worker, args=(session,), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    server_ref["server"] = server
    print(f"Phone page: http://10.225.167.142:{PORT}/", flush=True)

    def shutdown(signum, frame):
        stop.set()
        if server_ref["server"]:
            threading.Thread(target=server_ref["server"].shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
