# save as fix_to_activitynet.py
import json, os, sys

in_path = sys.argv[1]
video = os.path.basename(sys.argv[2]) if len(sys.argv) > 2 else "P01_webcam01_custom.mp4"

d = json.load(open(in_path))
anns = d.get("annotations", [])

# 将 “分钟” -> “秒”
def min_to_sec(x): return float(x) * 60.0

out = {"results": {video.rsplit(".", 1)[0]: []}}
for a in anns:
    seg = a.get("segment", [0,0])
    lab = a.get("label", "bg")
    out["results"][video.rsplit(".", 1)[0]].append({
        "label": lab,
        "segment": [min_to_sec(seg[0]), min_to_sec(seg[1])]
    })

json.dump(out, open(os.path.splitext(in_path)[0] + "_activitynet_sec.json","w"),
          ensure_ascii=False, indent=2)
print("Wrote:", os.path.splitext(in_path)[0] + "_activitynet_sec.json")
