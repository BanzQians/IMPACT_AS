import argparse, importlib, json, os
import numpy as np
import torch
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)   # 让 train.py 能找到 video_dataset 等顶层导入
sys.path.insert(0, ROOT)  # 让我们也能用 'src.train' 方式导入

def import_sym(spec: str):
    mod, sym = spec.split(":")
    return getattr(importlib.import_module(mod), sym)

def smart_build(model_ctor, ckpt):
    hp=None
    for k in ("hyper_parameters","hparams","config","args"):
        if isinstance(ckpt,dict) and k in ckpt and isinstance(ckpt[k],dict):
            hp=ckpt[k]; break
    try:
        return model_ctor(**(hp or {}))
    except TypeError:
        return model_ctor()

def smart_load_state_dict(model, ckpt_obj, strict=False):
    sd=ckpt_obj
    if isinstance(sd,dict) and "state_dict" in sd:
        sd=sd["state_dict"]
    def try_load(d):
        try:
            model.load_state_dict(d, strict=strict); return True
        except Exception:
            return False
    if try_load(sd): return
    for px in ("model.","module.","net."):
        d={ (k[len(px):] if k.startswith(px) else k):v for k,v in sd.items() }
        if try_load(d): return
    raise RuntimeError("failed to load_state_dict with common prefixes")

def segments_from_frame_labels(frame_labels, fps):
    segs=[]; 
    if not frame_labels: return segs
    cur=frame_labels[0]; s=0
    for i in range(1,len(frame_labels)):
        if frame_labels[i]!=cur:
            segs.append({"segment":[s/fps,i/fps],"label":cur}); cur=frame_labels[i]; s=i
    segs.append({"segment":[s/fps,len(frame_labels)/fps],"label":cur})
    return segs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)              # .ckpt 或 .pth
    ap.add_argument("--model", required=True)                # e.g. src.train:VideoSSL
    ap.add_argument("--features", required=True)             # T x D 的 .npy
    ap.add_argument("--class-names", required=True)          # 每行一个类名
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input-layout", default="BTD", choices=["BTD","BDT"], help="模型吃 BxTxD 还是 BxDxT")
    args=ap.parse_args()

    ck=torch.load(args.weights, map_location="cpu")
    ctor=import_sym(args.model)
    model=smart_build(ctor, ck).to(args.device).eval()

    feat=np.load(args.features)   # T x D
    x=torch.from_numpy(feat).float()
    if args.input_layout=="BTD":  # B x T x D
        x=x.unsqueeze(0)
    else:                          # B x D x T
        x=x.t().unsqueeze(0)
    x=x.to(args.device)

    smart_load_state_dict(model, ck, strict=False)

    with torch.no_grad():
        y=model(x)                 # 希望 -> (B,T,C) 或 (T,C) 或 (B,C,T)
    while y.dim()>2:
        y=y.squeeze(0)
    if y.shape[0]!=feat.shape[0] and y.shape[1]==feat.shape[0]:
        y=y.transpose(0,1)

    probs=torch.softmax(y, dim=-1)
    pred_idx=probs.argmax(dim=-1).cpu().numpy().tolist()

    with open(args.class_names, "r", encoding="utf-8") as f:
        classes=[ln.strip() for ln in f if ln.strip()]
    frame_labels=[ classes[i] if 0<=i<len(classes) else f"cls_{i}" for i in pred_idx ]

    segs=segments_from_frame_labels(frame_labels, args.fps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump({"annotations":segs}, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out} (frames={len(frame_labels)}, fps={args.fps})")

if __name__=="__main__":
    main()
