"""Evaluate on held-out test split (as the app does: crop-masked softmax), export ONNX, quantise, check parity,
pick blur threshold, and write model info + build assets for the app."""
import json, os, shutil, numpy as np, torch, timm
from PIL import Image
import torchvision.transforms as T
import onnxruntime as ort

S = json.load(open("training/split.json")); C = S["classes"]; NC = len(C)
A = "app/dist"; os.makedirs(A, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ev = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
m = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=NC)
m.load_state_dict(torch.load("training/best.pt", map_location="cpu")); m.eval()

test = [i for i in S["items"] if i["s"] == "test"]
crop_of = np.array([c.split("__")[0] for c in C])
logits = []
with torch.no_grad():
    for k in range(0, len(test), 64):
        x = torch.stack([ev(Image.open(i["p"]).convert("RGB")) for i in test[k:k+64]])
        logits.append(m(x).numpy())
logits = np.concatenate(logits); y = np.array([i["y"] for i in test])
pred_all = logits.argmax(1)
masked = logits.copy()
for n, i in enumerate(test):
    masked[n, crop_of != crop_of[i["y"]]] = -1e9
pred = masked.argmax(1)
ex = np.exp(masked - masked.max(1, keepdims=True)); prob = (ex / ex.sum(1, keepdims=True)).max(1)

rep = {"overall_unmasked": float((pred_all == y).mean()), "overall_crop_selected": float((pred == y).mean())}
crop_rows = []
for cr in ["paddy", "cotton", "maize", "chilli", "tomato"]:
    s = crop_of[y] == cr
    crop_rows.append({"crop": cr, "n": int(s.sum()), "acc": round(float((pred[s] == y[s]).mean()), 4)})
per_class = []
for ci, c in enumerate(C):
    s = y == ci
    if s.sum(): per_class.append({"c": c, "n": int(s.sum()), "acc": round(float((pred[s] == ci).mean()), 3),
                                  "top_confusion": C[int(np.bincount(pred[s & (pred != ci)], minlength=NC).argmax())] if (pred[s] != ci).any() else None})
hi = prob >= 0.70
rep["share_confident"] = float(hi.mean()); rep["acc_when_confident"] = float((pred[hi] == y[hi]).mean())
rep["acc_when_mid"] = float((pred[(prob >= .45) & ~hi] == y[(prob >= .45) & ~hi]).mean()) if ((prob >= .45) & ~hi).any() else None
rep["share_low"] = float((prob < .45).mean())
rep["per_crop"] = crop_rows; rep["per_class"] = per_class
rep["n_train"] = sum(1 for i in S["items"] if i["s"] == "train"); rep["n_test"] = len(test)
json.dump(rep, open("training/metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "per_class"}, indent=1))
for p in sorted(per_class, key=lambda r: r["acc"])[:10]: print(p)

# ---- ONNX export + dynamic int8 quantisation
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(m, dummy, "training/cropdoc_fp32.onnx", input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False,
                  dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}})
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
quant_pre_process("training/cropdoc_fp32.onnx", "training/cropdoc_pre.onnx")
quantize_dynamic("training/cropdoc_pre.onnx", "training/cropdoc_int8.onnx", weight_type=QuantType.QUInt8, per_channel=False)
def run(path, X):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); return s.run(None, {"input": X})[0]
X = torch.stack([ev(Image.open(i["p"]).convert("RGB")) for i in test[:400]]).numpy()
with torch.no_grad(): ref = m(torch.from_numpy(X)).numpy()
for name in ["fp32", "int8"]:
    out = run(f"training/cropdoc_{name}.onnx", X)
    print(name, "size MB", round(os.path.getsize(f"training/cropdoc_{name}.onnx") / 1e6, 2),
          "argmax agreement", float((out.argmax(1) == ref.argmax(1)).mean()), "max abs diff", float(np.abs(out - ref).max()))
