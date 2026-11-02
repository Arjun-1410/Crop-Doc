"""Assemble the publishable app folder: index.html (advice inlined), model, ORT runtime, sample + crop photos."""
import json, os, shutil, sys, numpy as np
from PIL import Image, ImageOps

MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "cropdoc_int8.onnx"
MODEL_OUT = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
S = json.load(open("training/split.json")); C = S["classes"]
R = json.load(open("training/metrics.json"))
D = "app/dist"; shutil.rmtree(D, ignore_errors=True)
for sub in ["ort", "samples", "crops"]: os.makedirs(f"{D}/{sub}", exist_ok=True)

# runtime + model
O = "node_modules/onnxruntime-web/dist"
for f in ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]: shutil.copy(f"{O}/{f}", f"{D}/ort/{f}")
shutil.copy(f"model/{MODEL_FILE}", f"{D}/{MODEL_OUT}")

def sq(src, dst, size, q=80):
    im = ImageOps.fit(Image.open(src).convert("RGB"), (size, size), Image.BICUBIC); im.save(dst, quality=q)

# blur threshold, same maths as the app (grey 0..1, 4-neighbour Laplacian variance on the 224 crop)
def lapvar(p):
    im = ImageOps.fit(Image.open(p).convert("RGB"), (224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255; g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    v = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(v.var())
tr = [i for i in S["items"] if i["s"] == "train"][::7]
lv = np.array([lapvar(i["p"]) for i in tr]); blur_min = float(np.percentile(lv, 1) * 0.5)
print("blur threshold", blur_min, "p1", np.percentile(lv, 1), "median", np.median(lv))

# example photos from the held-out test split (first image of each class, not hand-picked)
want = ["paddy__blast", "paddy__brown_spot", "paddy__dead_heart", "cotton__leaf_hopper_jassids", "cotton__leaf_redding",
        "maize__common_rust", "maize__blight", "chilli__murda_complex_mites_trips", "tomato__early_blight", "tomato__tomato_yellow_leaf_curl_virus"]
samples = []
for c in want:
    it = next(i for i in S["items"] if i["s"] == "test" and C[i["y"]] == c)
    fn = f"samples/{c}.jpg"; sq(it["p"], f"{D}/{fn}", 360, 82)
    samples.append({"file": fn, "crop": c.split("__")[0], "c": c})
for crop, c in [("paddy", "paddy__normal"), ("cotton", "cotton__healthy_leaf"), ("maize", "maize__healthy"), ("chilli", "chilli__healthy"), ("tomato", "tomato__healthy")]:
    it = next(i for i in S["items"] if i["s"] == "train" and C[i["y"]] == c)
    sq(it["p"], f"{D}/crops/{crop}.jpg", 180, 78)

info = {"file": MODEL_OUT, "bytes": os.path.getsize(f"{D}/{MODEL_OUT}"), "classes": C, "blurMin": round(blur_min, 6),
        "nTrain": R["n_train"], "nTest": R["n_test"], "metrics": R["per_crop"]}
html = open("app/index.src.html").read()
html = html.replace("/*ADVICE*/", open("app/advice.js").read())
html = html.replace("/*MODELINFO*/", json.dumps(info, ensure_ascii=False))
html = html.replace("/*SAMPLES*/", json.dumps(samples))
missing = [c for c in C if f'"{c}"' not in open("app/advice.js").read()]
assert not missing, missing
open(f"{D}/index.html", "w").write(html)
for root, _, fs in os.walk(D):
    for f in fs: print(os.path.relpath(os.path.join(root, f), D), os.path.getsize(os.path.join(root, f)))
