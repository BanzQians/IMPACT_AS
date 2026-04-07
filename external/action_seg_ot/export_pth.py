# export_pth.py
import argparse, os, glob, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt file or dir to pick latest/best")
    ap.add_argument("--out", default="final.pth")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        cands = sorted(glob.glob(os.path.join(ckpt_path, "*.ckpt")))
        assert cands, f"No .ckpt under {ckpt_path}"
        best = [p for p in cands if "best" in os.path.basename(p).lower()]
        ckpt_path = best[-1] if best else cands[-1]

    print("[INFO] loading:", ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    # 如需去前缀可在此处理：state = {k.replace("model.",""):v for k,v in state.items()}
    torch.save(state, args.out)
    print("[OK] saved:", args.out)

if __name__ == "__main__":
    main()
