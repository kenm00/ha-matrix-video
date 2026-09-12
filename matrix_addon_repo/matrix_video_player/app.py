#!/usr/bin/env python3
"""
Matrix Video Player — Home Assistant local add-on.

Fetches (and caches) a video via yt-dlp and streams it to a WLED matrix
over DDP, driven by simple HTTP endpoints:

  POST /play  {"video_url": "...", "cache_key": "optional_name"}
  POST /stop
  GET  /health

Reads its configuration (wled_host, matrix_width, matrix_height, fps)
from the add-on options at /data/options.json, which Supervisor
populates from whatever you set in the add-on's Configuration tab.
"""

import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from flask import Flask, jsonify, request

# ---- Load add-on options -------------------------------------------------
with open("/data/options.json") as f:
    OPTIONS = json.load(f)

WLED_HOST = OPTIONS.get("wled_host", "192.168.1.77")
MATRIX_WIDTH = int(OPTIONS.get("matrix_width", 32))
MATRIX_HEIGHT = int(OPTIONS.get("matrix_height", 32))
FPS = float(OPTIONS.get("fps", 20))

DDP_PORT = 4048
MAX_DATA_LEN = 1440  # bytes per DDP packet payload

CACHE_DIR = Path("/data/video_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_stream_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_seq = 0


def sanitize(name: str) -> str:
    """Turn an arbitrary string into a safe filename fragment."""
    name = (name or "").strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "_", name)
    return name or "untitled"


def send_ddp_frame(rgb_bytes: bytes) -> None:
    """Send one full frame to the WLED matrix, chunked into DDP packets."""
    global _seq
    total_len = len(rgb_bytes)
    offset = 0
    while offset < total_len:
        chunk = rgb_bytes[offset : offset + MAX_DATA_LEN]
        is_last = (offset + len(chunk)) >= total_len

        flags = 0x40  # protocol version 1
        if is_last:
            flags |= 0x01  # PUSH bit: display the frame once fully received

        _seq = (_seq + 1) % 16
        header = (
            bytes([flags, _seq, 0x01, 0x01])
            + offset.to_bytes(4, "big")
            + len(chunk).to_bytes(2, "big")
        )
        _sock.sendto(header + chunk, (WLED_HOST, DDP_PORT))
        offset += len(chunk)


def stream_loop(video_path: str) -> None:
    """Background thread: decode frames and push them to the matrix until stopped."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.error("Could not open video: %s", video_path)
        return

    frame_interval = 1.0 / FPS
    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop back to the start
                continue

            resized = cv2.resize(
                frame, (MATRIX_WIDTH, MATRIX_HEIGHT), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            send_ddp_frame(rgb.tobytes())
            time.sleep(frame_interval)
    finally:
        cap.release()


def stop_stream() -> None:
    global _stream_thread
    _stop_event.set()
    if _stream_thread and _stream_thread.is_alive():
        _stream_thread.join(timeout=5)
    _stream_thread = None
    _stop_event.clear()


def find_or_download(video_url: str, cache_key: str) -> Optional[Path]:
    """Return a cached local file for this video, downloading via yt-dlp if needed."""
    existing = list(CACHE_DIR.glob(f"{cache_key}.*"))
    if existing:
        app.logger.info("Cache hit: %s", cache_key)
        return existing[0]

    out_template = str(CACHE_DIR / f"{cache_key}.%(ext)s")
    app.logger.info("Cache miss — downloading: %s", video_url)
    result = subprocess.run(
        [
            "yt-dlp",
            video_url,
            "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "--merge-output-format", "mp4",
            "-o", out_template,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        app.logger.error("yt-dlp failed for %s:\n%s", video_url, result.stderr[-1500:])
        return None

    downloaded = list(CACHE_DIR.glob(f"{cache_key}.*"))
    return downloaded[0] if downloaded else None


@app.route("/play", methods=["POST"])
def play():
    global _stream_thread

    data = request.get_json(force=True, silent=True) or {}
    video_url = (data.get("video_url") or "").strip()
    cache_key = sanitize(data.get("cache_key") or video_url)

    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    video_path = find_or_download(video_url, cache_key)
    if video_path is None:
        return jsonify({"error": "could not find or download video"}), 502

    stop_stream()
    _stream_thread = threading.Thread(
        target=stream_loop, args=(str(video_path),), daemon=True
    )
    _stream_thread.start()

    return jsonify({"status": "playing", "file": str(video_path)})


@app.route("/stop", methods=["POST"])
def stop():
    stop_stream()
    return jsonify({"status": "stopped"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "wled_host": WLED_HOST})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787)
