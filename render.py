"""Warp artwork-space frames into projector space using calib/render_map.npz."""
import json
import os

import cv2
import numpy as np

from mapping import CALIB


class ProjectorWarp:
    def __init__(self):
        data = np.load(os.path.join(CALIB, "render_map.npz"))
        self.ref = int(data["ref"])
        x0, y0, x1, y1 = (int(v) for v in data["bbox"])
        self.offset = (x0, y0)
        # only the cover's bounding box is ever non-black, so frames are stored cropped to it
        self.map_x = np.ascontiguousarray(data["map_x"][y0:y1, x0:x1])
        self.map_y = np.ascontiguousarray(data["map_y"][y0:y1, x0:x1])

    def __call__(self, frame):
        """frame: REF x REF BGR image in artwork space -> cropped projector image."""
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def full(self, frame, size=(1920, 1080)):
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        patch = self(frame)
        x0, y0 = self.offset
        canvas[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch
        return canvas

    def write_meta(self, folder, fps, frames_per_beat=None):
        """meta.json tells projector.py where to blit and, for beat sync, the authored beat grid."""
        meta = {"offset": list(self.offset), "fps": fps}
        if frames_per_beat:
            meta["frames_per_beat"] = frames_per_beat
        with open(os.path.join(folder, "meta.json"), "w") as handle:
            json.dump(meta, handle)
