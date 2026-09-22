"""Register a clean reference image of the artwork to the camera's view and build the
projector lookup table.

    python register.py <reference image>

Chain, per projector pixel:  projector -> camera (structured light, smooth model)
                                       -> rectified camera view (corner homography)
                                       -> reference artwork (feature matches, smooth model)
Writes calib/render_map.npz: map_x/map_y (projector-sized, in REF x REF artwork pixels),
the covered bbox, and calib/reference.png (the artwork at REF x REF).
"""
import os
import sys

import cv2
import numpy as np

from mapping import CALIB, SIZE, apply_smooth, fit_smooth, load_mapping

REF = 1400
PROJ_W, PROJ_H = 1920, 1080


def match(reference, view):
    """Matched points (reference, view) using SIFT on contrast-normalised greys."""
    clahe = cv2.createCLAHE(3.0, (8, 8))
    grey = [clahe.apply(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in (reference, view)]
    grey[0] = cv2.GaussianBlur(grey[0], (0, 0), 2.0)  # the camera view is ~3x softer than the file
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02)
    (kp_r, des_r), (kp_v, des_v) = (sift.detectAndCompute(g, None) for g in grey)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_v, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_v[m.trainIdx].pt for m in good])
    _, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0)
    inliers = inliers.ravel().astype(bool)
    print(f"features: {len(kp_r)} ref / {len(kp_v)} view, {len(good)} matches, {inliers.sum()} on-plane")
    return src[inliers], dst[inliers]


def main():
    reference = cv2.resize(cv2.imread(sys.argv[1]), (REF, REF), interpolation=cv2.INTER_AREA)
    view = cv2.imread(os.path.join(CALIB, "artwork_cam.png"))
    mapping = load_mapping()

    ref_pts, view_pts = match(reference, view)
    view2ref, keep = fit_smooth(view_pts, ref_pts)
    error = np.linalg.norm(apply_smooth(view2ref, view_pts[keep]) - ref_pts[keep], axis=1)
    print(f"view->reference: {keep.sum()} points, rms {np.sqrt((error**2).mean()):.2f} artwork px"
          f" ({np.sqrt((error**2).mean()) * 960 / REF:.2f} projector px)")

    grid = np.stack(np.meshgrid(np.arange(PROJ_W, dtype=np.float32),
                                np.arange(PROJ_H, dtype=np.float32)), -1).reshape(-1, 2)
    cam = apply_smooth(mapping["proj2cam"], grid)
    rect = cv2.perspectiveTransform(cam[None].astype(np.float32), np.linalg.inv(mapping["H_ac"]))[0]
    art = apply_smooth(view2ref, rect).reshape(PROJ_H, PROJ_W, 2).astype(np.float32)

    # only trust the lookup where the structured light actually saw the cover
    hull = cv2.convexHull(mapping["corners_proj"].astype(np.float32))
    seen = np.zeros((PROJ_H, PROJ_W), np.uint8)
    cv2.fillConvexPoly(seen, hull.astype(np.int32), 255)
    seen = cv2.dilate(seen, np.ones((25, 25), np.uint8))
    inside = (art[..., 0] >= 0) & (art[..., 0] < REF) & (art[..., 1] >= 0) & (art[..., 1] < REF) & (seen > 0)
    ys, xs = np.nonzero(inside)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    covered = np.zeros((REF, REF), np.uint8)
    covered[art[inside][:, 1].astype(int), art[inside][:, 0].astype(int)] = 255
    covered = cv2.morphologyEx(covered, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    print(f"projector bbox {bbox}; beam reaches {covered.mean() / 255 * 100:.1f}% of the artwork"
          f" (rows {np.nonzero(covered.any(1))[0].min()}-{np.nonzero(covered.any(1))[0].max()} of {REF})")

    # the sleeve's real outline in artwork space (top/left/right are sleeve edges, bottom is the beam
    # limit), so animation can be masked to it instead of spilling onto whatever is behind
    t = np.linspace(0, SIZE, 60, dtype=np.float32)
    border = np.concatenate([np.stack([t, 0 * t], 1), np.stack([0 * t + SIZE, t], 1),
                             np.stack([t[::-1], 0 * t + SIZE], 1), np.stack([0 * t, t[::-1]], 1)])
    sleeve = np.zeros((REF, REF), np.uint8)
    cv2.fillPoly(sleeve, [apply_smooth(view2ref, border).round().astype(np.int32)], 255)
    cv2.imwrite(os.path.join(CALIB, "sleeve.png"), sleeve)

    art[~inside] = -1
    cv2.imwrite(os.path.join(CALIB, "reference.png"), reference)
    cv2.imwrite(os.path.join(CALIB, "covered.png"), covered)
    np.savez_compressed(os.path.join(CALIB, "render_map.npz"), map_x=art[..., 0], map_y=art[..., 1],
                        bbox=bbox, ref=REF)

    # side-by-side sanity image: camera view resampled into reference space vs the reference
    ref2view, _ = fit_smooth(ref_pts, view_pts)
    lookup = apply_smooth(ref2view, np.stack(np.meshgrid(np.arange(0, REF, dtype=np.float32),
                          np.arange(0, REF, dtype=np.float32)), -1).reshape(-1, 2)).reshape(REF, REF, 2)
    resampled = cv2.remap(view, lookup[..., 0].astype(np.float32), lookup[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(CALIB, "camera_in_ref_space.png"), resampled)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (0, 0), 1.5), 60, 140)
    check = resampled.copy()
    check[edges > 0] = (0, 255, 0)
    cv2.imwrite(os.path.join(CALIB, "register_check.jpg"), cv2.resize(check, (900, 900), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
