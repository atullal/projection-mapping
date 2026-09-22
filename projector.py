"""Projector display server (pygame only -- never import cv2 here).

Owns a borderless window covering the projector's display and takes line
commands over a localhost socket:

    show <image path>      display a still (calibration patterns, test outlines)
    play <dir> <fps>       loop the frames in <dir>; reads <dir>/meta.json for offset
    sync on|off            lock playback to the beat of whatever the microphone hears
    latency <ms>           how far ahead of the beat clock to draw (projector + HDMI lag)
    status                 tempo / confidence / level of the beat tracker
    black                  blank the projection
    quit

Frames are authored on a beat grid (meta.json frames_per_beat).  In sync mode the frame is
picked from the music's beat clock instead of the wall clock, so authored hits land on real
beats and the show speeds up or slows down with the song.

Every command is answered with "ok" once the new picture is actually on screen,
so the vision process can safely capture right after.  Esc blacks out.
"""
import glob
import json
import os
import socket
import sys
import time

import pygame

PORT = 47800


def main():
    display = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if display >= len(sizes):
        sys.exit(f"display {display} not found; desktops: {sizes}")
    size = sizes[display]
    screen = pygame.display.set_mode(size, pygame.NOFRAME, display=display)
    pygame.display.set_caption("Projection Mapper")
    pygame.mouse.set_visible(False)
    print(f"projector window {size} on display {display}", flush=True)

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.listen(4)
    server.setblocking(False)

    clock = pygame.time.Clock()
    frames, offset, fps, cursor = [], (0, 0), 30, 0
    still = None
    tracker, lead, frames_per_beat, beat_origin = None, 0.12, 18.0, 0.0

    def draw():
        screen.fill((0, 0, 0))
        if still is not None:
            screen.blit(still, (0, 0))
        elif frames:
            image = pygame.image.load(frames[cursor])
            if tracker is not None:                     # real onsets kick the brightness a little
                level = int(255 * (0.80 + 0.20 * tracker.flash))
                image.fill((level, level, level), special_flags=pygame.BLEND_RGB_MULT)
            screen.blit(image, offset)
        pygame.display.flip()

    def handle(line):
        nonlocal frames, offset, fps, cursor, still, tracker, lead, frames_per_beat, beat_origin
        cmd, _, arg = line.strip().partition(" ")
        if cmd == "show":
            image = pygame.image.load(arg).convert()
            still = image if image.get_size() == size else pygame.transform.smoothscale(image, size)
            frames = []
        elif cmd == "play":
            folder, _, rate = arg.rpartition(" ")
            frames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            if not frames:
                return f"error no frames in {folder}"
            meta_path = os.path.join(folder, "meta.json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            offset, fps, cursor, still = tuple(meta.get("offset", (0, 0))), float(rate), 0, None
            frames_per_beat = float(meta.get("frames_per_beat", 18.0))
            if tracker is not None:                     # start the show on the next beat
                beat_origin = float(int(tracker.beats()) + 1)
        elif cmd == "sync":
            if arg == "on" and tracker is None:
                from beat import BeatTracker            # numpy + sounddevice only: safe beside pygame
                tracker = BeatTracker()
                beat_origin = float(int(tracker.beats()) + 1) - cursor / frames_per_beat
            elif arg == "off" and tracker is not None:
                tracker.close()
                tracker = None
            return "ok"
        elif cmd == "latency":
            lead = float(arg) / 1000
            return "ok"
        elif cmd == "status":
            if tracker is None:
                return "ok sync=off"
            return (f"ok sync=on bpm={tracker.bpm:.1f} confidence={tracker.confidence:.2f} "
                    f"level={tracker.level:.2f} lead={lead * 1000:.0f}ms")
        elif cmd == "black":
            frames, still = [], None
        elif cmd == "quit":
            return None
        else:
            return f"error unknown command {cmd}"
        draw()
        pygame.event.pump()
        draw()  # second flip so both buffers hold the new picture
        return "ok"

    draw()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                frames, still = [], None
                draw()
        try:
            conn, _ = server.accept()
        except BlockingIOError:
            conn = None
        if conn:
            conn.setblocking(True)
            reply = handle(conn.makefile().readline())
            conn.sendall(((reply or "ok") + "\n").encode())
            conn.close()
            if reply is None:
                return
        if frames and tracker is not None:
            position = (tracker.beats(time.monotonic() + lead) - beat_origin) * frames_per_beat
            wanted = int(position) % len(frames)
            if wanted != cursor:
                cursor = wanted
                draw()
            clock.tick(120)
        elif frames:
            cursor = (cursor + 1) % len(frames)
            draw()
            clock.tick(fps)
        else:
            clock.tick(60)


if __name__ == "__main__":
    main()
