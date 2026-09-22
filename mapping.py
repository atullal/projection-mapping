"""Turn the dense calibration into a smooth mapping for one near-flat target (the album cover).

    python mapping.py <seed_x> <seed_y> [camera_index]

seed is any camera pixel on the cover.  A leaning record sleeve bows a little, so a plain
homography leaves a smooth ~5px error; the model here is homography + cubic residual.

Writes calib/mapping.npz (models both ways + cover corners) and the cover rectified to a
SIZE x SIZE square as calib/artwork_cam.png (projector-lit) / artwork_ambient.png.
"""
import os
import sys

import cv2
import numpy as np

from camera import grab, open_camera
from projector_client import send

SIZE = 1400
ROOT = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(ROOT, "calib")


def _terms(points, centre, scale):
    x, y = ((points - centre) / scale).T
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3], 1)


def fit_smooth(src, dst, rounds=4):
    """dst ~= H(src) + cubic(src), trimming outliers each round.  Returns (model, keep mask)."""
    keep = np.ones(len(src), bool)
    centre, scale = src.mean(0), src.std(0).mean() * 2
    for _ in range(rounds):
        H, _ = cv2.findHomography(src[keep], dst[keep], 0)
        base = cv2.perspectiveTransform(src[None].astype(np.float32), H)[0]
        coef, *_ = np.linalg.lstsq(_terms(src[keep], centre, scale), (dst - base)[keep], rcond=None)
        model = dict(H=H, coef=coef, centre=centre, scale=scale)
        error = np.linalg.norm(apply_smooth(model, src) - dst, axis=1)
        keep = error < max(3.0, 2.5 * np.sqrt((error[keep] ** 2).mean()))
    return model, keep


def apply_smooth(model, points):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    base = cv2.perspectiveTransform(points[None], model["H"])[0]
    return base + _terms(points, model["centre"], model["scale"]) @ model["coef"]


def cover_models(px, py, valid, seed, reach=420):
    """Fit the cover from decoded points near the seed; returns both models and the sleeve mask.

    Decoding can be patchy (dark inks in a dark room), so this does not need a contiguous
    blob: the plane fit picks its inliers and, the sleeve being convex, their hull is the mask.
    """
    window = np.zeros(valid.shape, bool)
    window[max(seed[1] - reach, 0):seed[1] + reach, max(seed[0] - reach, 0):seed[0] + reach] = True
    ys, xs = np.nonzero(valid & window)
    if len(xs) < 2000:
        raise SystemExit(f"only {len(xs)} decoded pixels near seed {seed}; is the beam on the cover?")
    cam = np.stack([xs, ys], 1).astype(np.float32)
    proj = np.stack([px[ys, xs], py[ys, xs]], 1).astype(np.float32)
    # coarse plane gate first so the background behind the cover doesn't steer the cubic
    _, coarse = cv2.findHomography(cam, proj, cv2.RANSAC, 10.0)
    coarse = coarse.ravel().astype(bool)
    cam2proj, _ = fit_smooth(cam[coarse], proj[coarse])
    # the coarse gate clips bowed corners; re-admit whatever the smooth model explains, refit once
    explained = np.linalg.norm(apply_smooth(cam2proj, cam) - proj, axis=1) < 5.0
    cam, proj = cam[explained], proj[explained]
    cam2proj, keep = fit_smooth(cam, proj)
    cam, proj = cam[keep], proj[keep]
    proj2cam, _ = fit_smooth(proj, cam)

    error = np.linalg.norm(apply_smooth(cam2proj, cam) - proj, axis=1)
    plain = np.linalg.norm(cv2.perspectiveTransform(cam[None], cam2proj["H"])[0] - proj, axis=1)
    print(f"cover points: {len(cam)}   rms homography only {np.sqrt((plain**2).mean()):.2f}px"
          f" -> with cubic {np.sqrt((error**2).mean()):.2f}px (4px stripe quantisation alone = 1.63)")

    dots = np.zeros(valid.shape, np.uint8)
    dots[cam[:, 1].astype(int), cam[:, 0].astype(int)] = 255
    dots = cv2.morphologyEx(dots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop lone chance inliers
    mask = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(cv2.findNonZero(dots)), 255)
    return cam2proj, proj2cam, mask


def fit_quad(mask):
    """Four corners (TL, TR, BR, BL) from line fits to each side of the mask's outline."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    rough = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float32)
    while len(rough) > 4:  # drop the corner whose removal loses the least area
        areas = [cv2.contourArea(np.delete(rough, i, 0)) for i in range(len(rough))]
        rough = np.delete(rough, int(np.argmax(areas)), 0)
    if len(rough) != 4:
        raise SystemExit(f"expected a quadrilateral, got {len(rough)} corners")
    centre = rough.mean(0)
    rough = rough[np.argsort(np.arctan2(rough[:, 1] - centre[1], rough[:, 0] - centre[0]))]
    rough = np.roll(rough, -int(np.argmin(rough.sum(1))), 0)  # start top-left, go clockwise

    points = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for a, b in zip(rough, np.roll(rough, -1, 0)):
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        along = (points - a) @ direction
        across = np.abs((points - a) @ np.array([-direction[1], direction[0]]))
        side = points[(along > 0.12 * length) & (along < 0.88 * length) & (across < 8)]
        vx, vy, x0, y0 = cv2.fitLine(side, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.cross([x0, y0, 1], [x0 + vx, y0 + vy, 1]))
    corners = [np.cross(lines[i - 1], lines[i]) for i in range(4)]
    return np.array([p[:2] / p[2] for p in corners], np.float32)


def load_mapping():
    data = np.load(os.path.join(CALIB, "mapping.npz"), allow_pickle=True)
    return {k: (data[k].item() if data[k].dtype == object else data[k]) for k in data.files}


def main():
    seed = (int(sys.argv[1]), int(sys.argv[2]))
    camera_index = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    data = np.load(os.path.join(CALIB, "cam2proj.npz"))
    cam2proj, proj2cam, mask = cover_models(data["px"], data["py"], data["valid"], seed)

    corners_cam = fit_quad(mask)
    square = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], np.float32)
    H_ac = cv2.getPerspectiveTransform(square, corners_cam)
    corners_proj = apply_smooth(cam2proj, corners_cam)
    print("corners (camera):   ", corners_cam.round(1).tolist())
    print("corners (projector):", corners_proj.round(1).tolist())

    cap = open_camera(camera_index)
    try:
        send(f"show {os.path.join(ROOT, 'patterns', 'white.png')}")
        lit = grab(cap, settle=1.2, average=8)
        send("black")
        ambient = grab(cap, settle=1.2, average=8)
    finally:
        cap.release()
    cv2.imwrite(os.path.join(CALIB, "scene_lit.png"), lit)
    cv2.imwrite(os.path.join(CALIB, "scene_ambient.png"), ambient)
    H_ca = np.linalg.inv(H_ac)
    for name, image in (("artwork_cam", lit), ("artwork_ambient", ambient)):
        cv2.imwrite(os.path.join(CALIB, name + ".png"),
                    cv2.warpPerspective(image, H_ca, (SIZE, SIZE), flags=cv2.INTER_CUBIC))

    overlay = lit.copy()
    cv2.polylines(overlay, [corners_cam.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(CALIB, "quad_debug.jpg"), overlay)
    np.savez(os.path.join(CALIB, "mapping.npz"), cam2proj=cam2proj, proj2cam=proj2cam, H_ac=H_ac,
             corners_cam=corners_cam, corners_proj=corners_proj, size=SIZE)


if __name__ == "__main__":
    main()
