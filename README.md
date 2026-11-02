# CropDoc AI

Snap a leaf photo. Get the crop disease and the cure in Telugu, in seconds.

CropDoc AI is an on-phone crop disease diagnosis app for farmers in the Hyderabad region of Telangana. A farmer picks the crop, photographs a leaf, and gets the disease name, a confidence score, Telugu text and voice advice, and the spray dose per litre, per 16 L tank and per field. The AI model runs inside the phone's browser, so the photo never leaves the phone.

**Team (Woxsen University):** Arjun Dosapati (24WU0102184), Ritesh Cheguri (24WU0102135), Sagar (4036), Vishwapal (2157), Kartheek (2154)
**Course:** Ideation and Business Model Design, Prof. Geeta Tripathi

## Results

Tested on 1,833 photos the model never saw during training (crop chosen first, as in the app).

| Crop | Test photos | Correct | Photo type |
|---|---|---|---|
| Tomato | 700 | 98.4% | Lab |
| Maize | 267 | 95.9% | Mostly lab |
| Cotton | 211 | 95.3% | Field, smartphone |
| Paddy | 600 | 87.5% | Field, India |
| Chilli | 55 | 80.0% | Field, India |
| **All** | **1,833** | **93.6%** | |

- When the model is confident (70% or more, 88% of photos) it is right **97.7%** of the time. Below 70% the app says "Possible" or "Not sure, ask an expert".
- Speed: 60 to 180 ms per photo in the browser.
- Full per-class results: [`training/metrics.json`](training/metrics.json).

## Features

- Crop picker: paddy, cotton, maize, chilli, tomato (36 diseases, pests and nutrient problems)
- Confidence bands, wrong-crop warning and dark or blurry photo warnings
- Telugu and English advice for every class, read aloud in Telugu
- Spray dose calculator, with doses taken from PJTAU and ICAR advisories (antibiotics left out, as India ordered them out of agriculture from 2024)
- Telangana season banner (Vanakalam, Yasangi) and local help: Kisan Call Centre 1800-180-1551, KVK Ranga Reddy (CRIDA Hayathnagar)
- "Was this right?" feedback, WhatsApp ask-expert message, scan history, CSV export

## Repository layout

| Path | What it is |
|---|---|
| `app/dist/` | Built app, ready to serve as a static site |
| `app/index.src.html`, `app/advice.js` | App source and the Telugu/English advice for all 36 classes |
| `model/cropdoc_fp32.onnx` | Trained model (input 1×3×224×224, ImageNet normalisation, 36 logits) |
| `training/` | Data prep, training, evaluation/export and build scripts, plus metrics and training log |
| `docs/` | Business model canvas, Prototype Readiness Report, pitch deck content |

## Run the app

```bash
cd app/dist
python -m http.server 8000
# open http://localhost:8000 on your computer or phone (same Wi-Fi)
```

`app/dist/model.wasm` is the same ONNX model as `model/cropdoc_fp32.onnx`, renamed so static hosts serve it.

## Retrain the model

```bash
pip install -r requirements.txt
# 1. Download datasets into data/repos/ (see Data below)
# 2. Download ImageNet weights:
mkdir -p weights && curl -L -o weights/mobilenetv3_large_100_ra-f55367f5.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
python training/prep.py      # split + 256 px cache
python training/train.py 4   # 4 epochs, about 1 hour on 2 CPU cores
python training/export.py    # test-set evaluation + ONNX export
npm install onnxruntime-web@1.29.0
cp training/cropdoc_fp32.onnx model/
python training/build.py cropdoc_fp32.onnx model.wasm   # rebuild app/dist
```

Model: MobileNetV3-Large, ImageNet-pretrained (timm), fine-tuned for 4 epochs with crop, flip, rotation and colour augmentation. Images are capped at 700 per class and split 80/10/10 per class.

## Data

| Crop | Dataset |
|---|---|
| Paddy | [Paddy Doctor](https://github.com/ai-agriculture-circuits-and-systems/paddy_disease_classification) (Indian field photos) |
| Cotton | [SAR-CLD-2024](https://github.com/poojashreer123/smart-cotton-disease-detection) (smartphone field photos) |
| Maize | [Corn/maize leaf disease](https://github.com/ai-agriculture-circuits-and-systems/corn_maize_leaf_disease) |
| Tomato | [PlantVillage tomato](https://github.com/ai-agriculture-circuits-and-systems/tomato_leaf_disease) |
| Chilli | [Chilli leaf classification](https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification) |

## Limitations

- Not yet tested on Telangana farm photos or with farmers; field accuracy will be lower than the numbers above.
- Paddy (87.5%) and chilli (80.0%) need more local data; the chilli leaf-curl complex is only 50%.
- Tomato and maize scores are likely optimistic (lab photos with plain backgrounds).
- Advice must be reviewed by a KVK or PJTAU scientist before public use.
- No "not a leaf" class; first load downloads about 31 MB.

## Built with

PyTorch, timm, ONNX Runtime Web, and the public datasets above. Development and research were assisted by Claude (Anthropic). This is a university prototype; it is not a substitute for advice from an agriculture officer.
