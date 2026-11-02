"""Build class list, stratified split and a 256px image cache for CropDoc training."""
import os, json, random, glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps

R = os.environ.get("CROPDOC_DATA", "data/repos")
OUT = os.environ.get("CROPDOC_CACHE", "data/cache")
CAP = 700          # max images per class (keeps CPU training time reasonable, reduces imbalance)
random.seed(42)

SRC = {
  "paddy":  (f"{R}/ai-agriculture-circuits-and-systems_paddy_disease_classification/paddies", "{c}/images"),
  "cotton": (f"{R}/poojashreer123_smart-cotton-disease-detection/dataset/Original Dataset", "{c}"),
  "maize":  (f"{R}/ai-agriculture-circuits-and-systems_corn_maize_leaf_disease/corn_leaves", "{c}/images"),
  "tomato": (f"{R}/ai-agriculture-circuits-and-systems_tomato_leaf_disease/tomato_leaves", "{c}/images"),
  "chilli": (f"{R}/ai-agriculture-circuits-and-systems_chilli_leaf_classification/chillies", "{c}/images"),
}
def slug(s): return s.lower().replace(" ", "_").replace("-", "_")

def resize(job):
    src, dst = job
    if os.path.exists(dst): return True
    try:
        im = Image.open(src); im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((10_000, 10_000))
        w, h = im.size; s = 256 / min(w, h)
        im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, quality=92)
        return True
    except Exception as e:
        print("bad", src, e); return False

classes, items, jobs = [], [], []
for crop, (root, pat) in SRC.items():
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, pat.format(c=c))
        if not os.path.isdir(d): continue
        files = sorted(f for f in glob.glob(d + "/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(files) < 20: continue
        random.shuffle(files); files = files[:CAP]
        name = f"{crop}__{slug(c)}"
        ci = len(classes); classes.append(name)
        n = len(files); nv = max(10, int(n * .1)); nt = max(10, int(n * .1))
        for i, f in enumerate(files):
            split = "test" if i < nt else "val" if i < nt + nv else "train"
            dst = f"{OUT}/{name}/{i:05d}.jpg"
            jobs.append((f, dst)); items.append({"p": dst, "y": ci, "s": split})

print(len(classes), "classes,", len(items), "images")
with ProcessPoolExecutor(2) as ex:
    ok = list(ex.map(resize, jobs, chunksize=64))
items = [it for it, k in zip(items, ok) if k]
json.dump({"classes": classes, "items": items}, open("training/split.json", "w"))
from collections import Counter
cnt = Counter((classes[i["y"]], i["s"]) for i in items)
for c in classes: print(f"{c:45s} train {cnt[(c,'train')]:4d} val {cnt[(c,'val')]:3d} test {cnt[(c,'test')]:3d}")
