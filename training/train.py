"""Fine-tune ImageNet MobileNetV3-Large on the CropDoc Hyderabad crop set (CPU)."""
import json, time, math, torch, timm, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

torch.manual_seed(0); torch.set_num_threads(2)
S = json.load(open("training/split.json")); C = S["classes"]; NC = len(C)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.45, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                   T.RandomRotation(20), T.ColorJitter(0.35, 0.35, 0.3, 0.04), T.RandomGrayscale(0.03),
                   T.ToTensor(), T.Normalize(MEAN, STD)])
ev_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

class D(Dataset):
    def __init__(s, split, tf): s.it = [i for i in S["items"] if i["s"] == split]; s.tf = tf
    def __len__(s): return len(s.it)
    def __getitem__(s, k): i = s.it[k]; return s.tf(Image.open(i["p"]).convert("RGB")), i["y"]

tr, va = D("train", tr_tf), D("val", ev_tf)
counts = np.bincount([i["y"] for i in tr.it], minlength=NC)
w = torch.tensor((counts.mean() / counts) ** 0.5, dtype=torch.float32)   # soften imbalance (chilli/cotton are small)
tl = DataLoader(tr, 32, shuffle=True, num_workers=1, drop_last=True, persistent_workers=True)
vl = DataLoader(va, 64, num_workers=1)

m = timm.create_model("mobilenetv3_large_100", pretrained=False)
m.load_state_dict(torch.load("weights/mobilenetv3_large_100_ra-f55367f5.pth", map_location="cpu"))
m.reset_classifier(NC)
frozen = [m.conv_stem, m.bn1, *m.blocks[:3]]
for mod in frozen:
    for p in mod.parameters(): p.requires_grad = False
params = [p for p in m.parameters() if p.requires_grad]
EP = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 4
steps = EP * len(tl)
opt = torch.optim.AdamW(params, lr=1.5e-3, weight_decay=0.02)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.12)
lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

def evaluate():
    m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in vl: ok += (m(x).argmax(1) == y).sum().item(); n += len(y)
    return ok / n

best = 0; t0 = time.time()
for ep in range(EP):
    m.train()
    for mod in frozen: mod.eval()          # keep frozen BN stats fixed
    tot = 0
    for k, (x, y) in enumerate(tl):
        loss = lossf(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if k % 50 == 0: print(f"ep{ep} it{k}/{len(tl)} loss {tot/(k+1):.3f} {time.time()-t0:.0f}s", flush=True)
    acc = evaluate(); print(f"== epoch {ep} val acc {acc:.4f} time {time.time()-t0:.0f}s", flush=True)
    if acc > best: best = acc; torch.save(m.state_dict(), "training/best.pt")
print("BEST", best)
