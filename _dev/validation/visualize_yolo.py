"""Overlay YOLO segmentation labels on source frames and write an MP4 video."""

import argparse
import glob
import os

import cv2
import numpy as np
import yaml


def parse_yolo_label(label_path, img_w, img_h):
    """Parse a YOLO segmentation label file into class ids and pixel polygons."""
    objects = []
    if not os.path.exists(label_path):
        return objects
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords) - 1, 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                pts.append([x, y])
            if pts:
                objects.append((cls_id, np.array(pts, dtype=np.int32)))
    return objects


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation on frames.")
    parser.add_argument("--yolo-dir", required=True, help="Path to formatYOLO directory")
    parser.add_argument("--output", default="yolo_overlay.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (default: read from run_summary)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency")
    args = parser.parse_args()

    images_dir = os.path.join(args.yolo_dir, "images", "train")
    labels_dir = os.path.join(args.yolo_dir, "labels", "train")
    data_yaml = os.path.join(args.yolo_dir, "data.yaml")

    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", {})

    # Brand colors: Garnet for wet, Atlantic for dry
    class_colors = {
        0: (86, 106, 70),   # dry  -> Atlantic BGR
        1: (10, 0, 115),    # wet  -> Garnet BGR
    }

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    sample = cv2.imread(image_files[0])
    h, w = sample.shape[:2]

    fps = args.fps
    if fps is None:
        summary_path = os.path.join(os.path.dirname(args.yolo_dir), "run_summary.yaml")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
            fps = summary.get("video_fps", 10.0)
        else:
            fps = 10.0

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for img_path in image_files:
        frame = cv2.imread(img_path)
        overlay = frame.copy()

        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        objects = parse_yolo_label(label_path, w, h)

        for cls_id, polygon in objects:
            color = class_colors.get(cls_id, (0, 255, 0))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(frame, [polygon], True, color, 2)

        frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

        # Draw bounding boxes
        for cls_id, polygon in objects:
            color = class_colors.get(cls_id, (0, 255, 0))
            x, y, bw, bh = cv2.boundingRect(polygon)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
            label = str(class_names.get(cls_id, cls_id))
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Legend
        y_off = 30
        for cid, name in sorted(class_names.items()):
            color = class_colors.get(cid, (0, 255, 0))
            cv2.rectangle(frame, (10, y_off - 15), (30, y_off), color, -1)
            cv2.putText(frame, str(name), (35, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_off += 30

        writer.write(frame)

    writer.release()
    print(f"Wrote {len(image_files)} frames to {args.output}")


if __name__ == "__main__":
    main()
