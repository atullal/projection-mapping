"""Regenerate the static projector patterns (white, labelled grid).  Gray-code stripes are
made on demand by calibrate.py.

    python make_patterns.py
"""
import os

import cv2
import numpy as np

W, H = 1920, 1080
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patterns")


def main():
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "white.png"), np.full((H, W, 3), 255, np.uint8))

    # labelled grid: read projector coordinates straight off a camera photo (cell = 160 x 135 px)
    grid = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W + 1, 160):
        cv2.line(grid, (min(x, W - 1), 0), (min(x, W - 1), H), (0, 255, 0), 3)
    for y in range(0, H + 1, 135):
        cv2.line(grid, (0, min(y, H - 1)), (W, min(y, H - 1)), (0, 255, 0), 3)
    for x in range(0, W, 160):
        for y in range(0, H, 135):
            cv2.putText(grid, f"{x // 160},{y // 135}", (x + 12, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.rectangle(grid, (2, 2), (W - 3, H - 3), (0, 0, 255), 6)
    cv2.imwrite(os.path.join(OUT, "grid.png"), grid)
    print(f"wrote white.png and grid.png to {OUT}")


if __name__ == "__main__":
    main()
