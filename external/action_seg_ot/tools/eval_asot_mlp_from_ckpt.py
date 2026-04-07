import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

# 可选：修正路径（如果不作为包运行时）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

def build_mlp_from_hparams(hp: dict, state_dict: dict) -> nn.Module:
    """根据 ckpt 里的 layer_sizes 构建一个与训练时一致的 MLP 结构。
    兼容两种命名：
      - 'mlp.0.0.weight' 形式（子Sequential里套Linear）
      - 'mlp.1.weight' 形式（直接Linear）
    """
    sizes = hp.get("layer_sizes", None)
    if not sizes or len(sizes) < 2:
        raise ValueError(f"Invalid layer_sizes in hyper_parameters: {sizes}")
    layers = nn.ModuleList()

    # 观察 state_dict 的键，判断第一层是否用到了 'mlp.0.0.'
    has_nested0 = any(k.startswith("mlp.0.0.") for k in state_dict.keys())

    # 构建线性层序列（中间层后接 ReLU，最后一层不接激活）
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i+1])
        if i < len(sizes) - 2:
            block = nn.Sequential(lin, nn.ReLU())
        else:
            block = lin
        layers.append(block)

    # 用一个顶层 Sequential 包起来，并赋名字以尽量对齐 state_dict
    seq = nn.Sequential()
    for i, block in enumerate(layers):
        if i == 0 and has_nested0:
            # 训练时第一层可能是 mlp.0.0 (Linear) + mlp.0.1 (ReLU)
            seq.add_module("0", block)  # block 是 Sequential(Linear, ReLU)
        else:
            seq.add_module(str(i if not has_nested0 else i), block)

    # 再包一层，命名为 'mlp'，这样 keys 形如 'mlp.0.0.weight'
    model = nn.Sequential()
    model.add_module("mlp", seq)
    return model

def load_mlp_weights(model: nn.Module, ckpt_obj):
    """从 ckpt 对象里提取 mlp.* 权重并加载到 model"""
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        sd = ckpt_obj["state_dict"]
    else:
        sd = ckpt_obj

    # 只保留 mlp.* 的键
    mlp_sd = { k: v for k, v in sd.items() if k.startswith("mlp.") }
    # 也兼容 'module.mlp.' 前缀
    if not mlp_sd:
        mlp_sd = { k.replace("module.", "", 1): v for k, v in sd.items() if k.startswith("module.mlp.") }

    if not mlp_sd:
        raise KeyError("No 'mlp.' keys found in state_dict. Available keys sample: "
                       + ", ".join(list(sd.keys())[:10]))

    # 尝试严格加载；失败则放宽
    try:
        model.load_state_dict(mlp_sd, strict=True)
    except Exception:
        model.load_state_dict(mlp_sd, strict=False)

def merge_frame_labels_to_segments(frame_labels, fps: float):
    segs = []
    if not frame_labels:
        return segs
    cur = frame_labels[0]; s = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != cur:
            segs.append({"segment":[s/fps, i/fps], "label": cur})
            cur = frame_labels[i]; s = i
    segs.append({"segment":[s/fps, len(frame_labels)/fps], "label": cur})
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".ckpt 或 .pth（包含 state_dict）")
    ap.add_argument("--features", required=True, help="T x D 的 .npy（逐帧特征）")
    ap.add_argument("--class-names", required=True, help="txt，每行一个类别名")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # 1) 载 ckpt
    ck = torch.load(args.weights, map_location="cpu")
    hp = None
    if isinstance(ck, dict):
        hp = ck.get("hyper_parameters") or ck.get("hparams") or ck.get("config") or ck.get("args")
    if not isinstance(hp, dict):
        raise ValueError("hyper_parameters not found in ckpt; cannot infer layer_sizes.")

    # 2) 根据超参构建 MLP，并加载权重
    model = build_mlp_from_hparams(hp, ck["state_dict"] if isinstance(ck, dict) else ck)
    load_mlp_weights(model, ck)
    model.to(args.device).eval()

    # 3) 读特征（T x D）并校验输入维度
    feat = np.load(args.features)      # T x D
    T, D = int(feat.shape[0]), int(feat.shape[1])
    input_dim = hp["layer_sizes"][0]
    if D != input_dim:
        raise ValueError(f"Feature dim {D} != expected input_dim {input_dim}. "
                         f"请确认特征是否与训练时对齐（可能需先做 PCA 投影）。")

    x = torch.from_numpy(feat).float().to(args.device)  # T x D
    with torch.no_grad():
        logits = model.mlp[0](x) if isinstance(model.mlp, nn.Sequential) else model(x)  # 逐帧
        # 兼容：如果上一行不适配，就直接 model(x)
        if not isinstance(logits, torch.Tensor):
            logits = model(x)

    # 4) 取类别名并做 argmax
    with open(args.class_names, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1).cpu().numpy().tolist()
    frame_labels = [ classes[i] if 0 <= i < len(classes) else f"cls_{i}" for i in idx ]

    # 5) 合段 → ActivityNet 段级 JSON
    segs = merge_frame_labels_to_segments(frame_labels, args.fps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"annotations": segs}, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out} (T={T}, D={D}, fps={args.fps})")

if __name__ == "__main__":
    main()
