"""First thing to run on a new machine or after re-plugging anything.

    python check_rig.py

Lists displays, tries every camera index and saves a probe photo of each, and lists
microphones -- so you can tell which index sees the album and which display is the
projector before touching the pipeline.  Also triggers the macOS camera/mic permission
dialogs, which must be accepted for whatever terminal app runs this.
"""
import os
import subprocess
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))


def displays():
    print("== displays (macOS) ==")
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=20).stdout
    except Exception as error:  # not macOS, or system_profiler missing
        print(f"  (could not query: {error})")
        return
    for line in out.splitlines():
        if line.startswith("        ") and line.strip().endswith(":") and not line.startswith("          "):
            print("  " + line.strip().rstrip(":"))
        elif "Resolution:" in line or "Main Display:" in line:
            print("     " + line.strip())
    print("  projector.py takes a pygame display index: 0 is the main display, 1 is usually the projector")


def cameras(max_index=4):
    print("== cameras ==")
    os.makedirs(os.path.join(ROOT, "captures"), exist_ok=True)
    found = 0
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else 0)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        deadline = time.time() + 1.5
        while time.time() < deadline:
            cap.grab()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  index {index}: opened but returned no frame")
            continue
        path = os.path.join(ROOT, "captures", f"probe_cam{index}.jpg")
        cv2.imwrite(path, frame)
        print(f"  index {index}: {frame.shape[1]}x{frame.shape[0]}  mean brightness {frame.mean():.0f}  -> {path}")
        found += 1
    if not found:
        print("  no camera opened. On macOS: System Settings > Privacy & Security > Camera, allow your terminal app.")
    else:
        print("  LOOK AT THE PROBE IMAGES: pass the index that shows the album to calibrate.py / mapping.py")


def microphones():
    print("== microphones ==")
    try:
        import sounddevice as sd
    except ImportError:
        print("  sounddevice not installed (only needed for music sync)")
        return
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            default = " (default)" if index == sd.default.device[0] else ""
            print(f"  {index}: {device['name']}{default}")
    print("  beat.py opens the input by NAME (BeatTracker(device=...)); default is 'MacBook Pro Microphone'")


if __name__ == "__main__":
    displays()
    cameras()
    microphones()
