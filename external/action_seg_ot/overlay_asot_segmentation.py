#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay ASOT (action segmentation) predictions onto a video for visual inspection.

Usage:
  python overlay_asot_segmentation.py \
    --video /path/to/input.mp4 \
    --pred /path/to/pred.txt \
    --mapping /path/to/mapping.txt \
    --out /path/to/out_demo.mp4 \
    --label_fps 15 --timeline --legend
"""

import argparse
import cv2
import numpy as np
import os, json, csv, math, re
from collections import defaultdict


def read_mapping(mapping_path):
    id2name, name2id = {}, {}
    if mapping_path is None: return id2name, name2id
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0].isdigit():
                idx, name = int(parts[0]), " ".join(parts[1:])
            elif parts[-1].isdigit():
                idx, name = int(parts[-1]), " ".join(parts[:-1])
            else:
                continue
            id2name[idx] = name
            name2id[name] = idx
    return id2name, name2id


def read_predictions(pred_path):
    ext = os.path.splitext(pred_path)[1].lower()
    if ext == '.json':
        with open(pred_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "frame_labels" in data:
            return {"frame_labels": data["frame_labels"]}
        if "segments" in data:
            out = []
            for s in data["segments"]:
                out.append({
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", 0)),
                    "label": s.get("label"),
                    "units": s.get("units", "seconds")
                })
            return {"segments": out}
    elif ext == '.csv':
        segments = []
        with open(pred_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = [c.lower() for c in reader.fieldnames]
            mode = 'frames' if 'start_frame' in cols else 'seconds'
            for row in reader:
                start = float(row.get('start_frame', row.get('start', 0)))
                end = float(row.get('end_frame', row.get('end', start)))
                label = row.get('label')
                segments.append({
                    "start": start, "end": end, "label": label,
                    "units": 'frames' if mode == 'frames' else 'seconds'
                })
        return {"segments": segments}
    else:
        with open(pred_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) == 1 and (" " in lines[0] or "\t" in lines[0]):
            tokens = re.split(r"\s+", lines[0])
        else:
            tokens = []
            for ln in lines:
                parts = re.split(r"\s+", ln)
                tokens.extend([p for p in parts if p])
        return {"frame_labels": tokens}


def map_to_names(labels, id2name):
    out = []
    for lb in labels:
        if isinstance(lb, (int, np.integer)) or (isinstance(lb, str) and lb.isdigit()):
            out.append(id2name.get(int(lb), str(lb)))
        else:
            out.append(str(lb))
    return out


def segments_to_frame_labels(segments, total_frames, fps, id2name):
    frame_labels = [""] * total_frames
    for seg in segments:
        label = seg["label"]
        if isinstance(label, (int, np.integer)) or (isinstance(label, str) and label.isdigit()):
            label = id2name.get(int(label), str(label))
        start_frame = int(round(seg["start"] * fps)) if seg.get("units") == "seconds" else int(seg["start"])
        end_frame = int(round(seg["end"] * fps)) if seg.get("units") == "seconds" else int(seg["end"])
        for fr in range(max(0, start_frame), min(total_frames, end_frame)):
            frame_labels[fr] = label
    last = ""
    for i in range(total_frames):
        if frame_labels[i] == "":
            frame_labels[i] = last
        else:
            last = frame_labels[i]
    return frame_labels


def upsample_labels(frame_labels, total_frames, label_fps, video_fps):
    ratio = float(video_fps) / float(label_fps)
    out = []
    for i in range(total_frames):
        src = int(round(i / ratio))
        if src >= len(frame_labels): src = len(frame_labels) - 1
        out.append(frame_labels[src])
    return out


def make_palette(names):
    rng = np.random.default_rng(2024)
    base_colors = rng.integers(0, 255, size=(max(12, len(set(names))), 3), dtype=np.uint8)
    cmap = {}
    for i, n in enumerate(sorted(set(names))):
        cmap[n] = tuple(int(c) for c in base_colors[i % len(base_colors)])
    for k in list(cmap.keys()):
        if k.lower() in ("background", "bg", "", "none"):
            cmap[k] = (40, 40, 40)
    return cmap


def draw_timeline(width, height, labels, cmap, bar_h=10):
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
    n = len(labels)
    for x in range(width):
        idx = int((x / max(1, width - 1)) * (n - 1))
        bar[:, x, :] = cmap.get(labels[idx], (128, 128, 128))
    return bar


def overlay_text(frame, text, alpha=0.35, pad=8, color=(255,255,255), thickness=2):
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.8, thickness)
    box_w, box_h = tw + 2*pad, th + base + 2*pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (box_w, box_h), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    cv2.putText(frame, text, (pad, pad+th), font, 0.8, color, thickness, cv2.LINE_AA)
    return frame


def add_legend(frame, cmap):
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sw = 18; margin = 10; col_w = 220; max_cols = 3
    names = sorted(cmap.keys())
    rows = math.ceil(len(names)/max_cols)
    legend_w = min(max_cols, math.ceil(len(names)/rows))*col_w + margin*2
    legend_h = rows*(sw+8) + margin*2
    x0 = max(0, W - legend_w - margin)
    y0 = margin
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+legend_w, y0+legend_h), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    x, y, col = x0+margin, y0+margin+sw, 0
    for i, n in enumerate(names):
        color = cmap[n]
        cv2.rectangle(frame, (x, y-sw), (x+sw, y), color, -1)
        cv2.putText(frame, n[:24], (x+sw+8, y-4), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        col += 1
        if col >= max_cols:
            col, x = 0, x0+margin
            y += sw + 8
        else:
            x += col_w
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--mapping', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label_fps', type=float, default=None)
    ap.add_argument('--alpha', type=float, default=0.35)
    ap.add_argument('--thickness', type=int, default=2)
    ap.add_argument('--legend', action='store_true')
    ap.add_argument('--timeline', action='store_true')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {args.video}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    id2name, name2id = read_mapping(args.mapping)
    pred = read_predictions(args.pred)

    if "frame_labels" in pred:
        names = map_to_names(pred["frame_labels"], id2name)
        if args.label_fps and abs(args.label_fps - vid_fps) > 1e-3:
            labels = upsample_labels(names, total_frames, args.label_fps, vid_fps)
        elif len(names) != total_frames:
            labels = [names[int(round(i*(len(names)-1)/max(1,total_frames-1)))] for i in range(total_frames)]
        else:
            labels = names
    else:
        labels = segments_to_frame_labels(pred["segments"], total_frames, vid_fps, id2name)

    cmap = make_palette(labels)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, vid_fps, (W, H))
    tl_bar = draw_timeline(W, H, labels, cmap) if args.timeline else None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        label = labels[min(frame_idx, len(labels)-1)]
        t = frame_idx / max(1e-6, vid_fps)
        text = f"{label} | t={t:6.2f}s ({frame_idx+1}/{total_frames})"
        frame = overlay_text(frame, text, alpha=args.alpha, thickness=args.thickness)
        color = cmap.get(label, (128,128,128))
        cv2.rectangle(frame, (5,5), (25,25), color, -1)
        if tl_bar is not None:
            y0 = H - tl_bar.shape[0]
            frame[y0:H, 0:W, :] = tl_bar
            x_cursor = int((frame_idx / max(1, total_frames-1)) * (W-1))
            cv2.line(frame, (x_cursor, H-tl_bar.shape[0]), (x_cursor, H-1), (255,255,255), 1)
        if args.legend and (frame_idx % int(vid_fps) == 0):
            frame = add_legend(frame, cmap)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
