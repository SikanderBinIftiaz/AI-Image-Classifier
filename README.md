# VisionAI – AI Image Classification System

A complete image classification application built with **PyTorch (CNN)**, **Streamlit**, **OpenCV** and **SQLite**.
Upload an image, get the predicted category with a confidence score, track your history, evaluate models, train on your own dataset, use your webcam and compare several networks.

## Task checklist

| Requirement | Where it lives |
|---|---|
| Image Upload | **Classify** page (`views/classify.py`), single or multiple files |
| Image Classification | `src/models.py` – pretrained CNNs (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0) |
| Prediction Confidence | Confidence meter, level badge and top-K chart (`src/ui.py`) |
| Result Dashboard | **Dashboard** page (`views/dashboard.py`) |
| History Tracking | **History** page + SQLite database (`src/database.py`) |
| Model Evaluation | **Evaluate** page – accuracy, precision, recall, F1, top-5, confusion matrix (`src/evaluation.py`) |
| Upgrade: Custom Dataset Training | **Train** page – transfer learning on your folders (`src/training.py`) |
| Upgrade: Real-Time Camera Detection | **Camera** page (snapshot and live stream) and `realtime_camera.py` |
| Upgrade: Multiple Model Comparison | **Compare** page (`views/compare.py`) |

---

## Step-by-step: run the project

### Step 1 – Install Python
Install **Python 3.10 or newer** (3.11 or 3.12 recommended) from python.org. On Windows tick **"Add Python to PATH"**.
Check it worked:
```bash
python --version
```

### Step 2 – Open a terminal in the project folder
Unzip the project, then open a terminal (Command Prompt, PowerShell or Terminal) inside the `ai-image-classifier` folder.

### Step 3 – Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 4 – Install the libraries
```bash
pip install -r requirements.txt
```
This downloads PyTorch, so it can take a few minutes. (For NVIDIA GPU support, pick the matching command at pytorch.org. The CPU version works fine for this project.)

### Step 5 – Check that everything works (optional but useful)
```bash
python smoke_test.py
```
It trains a tiny model on generated images, evaluates it, classifies an image and tests the database. You should see `ALL CHECKS PASSED`. It needs no internet.

### Step 6 – Start the app
```bash
streamlit run app.py
```
The app opens at http://localhost:8501. The **first** classification downloads the pretrained weights (14 to 100 MB depending on the model) and needs internet once.

### Step 7 – Try every feature
1. **Classify** – drop in a photo of a dog, a car, a coffee mug, and so on. Change the model or the number of predictions.
2. **Dashboard** and **History** – fill up after a few predictions.
3. Create the sample dataset for the upgrade features (needs internet once, about 170 MB):
   ```bash
   python make_sample_dataset.py
   ```
4. **Train** – choose *Sample dataset*, keep the defaults (MobileNetV2, 5 epochs) and press **Start training**. Your model is saved in `saved_models/`.
5. **Evaluate** – choose *Custom: my_classifier* and the folder `data/sample_dataset/test` (already filled in). You get accuracy, precision, recall, F1 and the confusion matrix.
6. **Compare** – upload one image and run several models side by side.
7. **Camera** – *Snapshot* uses your browser camera. *Live stream* runs continuously on your webcam.

### Step 8 – Real-time detection in its own window (smoothest)
```bash
python realtime_camera.py
python realtime_camera.py --model ResNet18
python realtime_camera.py --model "Custom: my_classifier"
```
Keys: `q` or `Esc` quits, `m` switches model, `s` saves a screenshot to `data/screenshots/`.

---

## Using your own dataset

Create one folder per category and put that category's images inside:

```
my_dataset/
├── cats/    (cat1.jpg, cat2.jpg, ...)
├── dogs/
└── birds/
```

On the **Train** page choose *Folder on this computer* (paste the path) or *Upload a ZIP file* of this folder. Aim for **50 or more images per category**. For evaluation, prepare a separate `test` folder in the same layout with images the model did not train on.

Tip: if your dataset has `train/` and `test/` folders, point the app at the parent folder. It automatically uses `train/` for training.

---

## Project structure and build order

The order below is also a good order to write the code yourself, from the foundation upward.

```
ai-image-classifier/
├── app.py                  9. Entry point: page setup, sidebar navigation
├── requirements.txt        Libraries to install
├── smoke_test.py           Automated end-to-end check
├── make_sample_dataset.py  Builds a small practice dataset (CIFAR-10 subset)
├── realtime_camera.py      8. Live webcam window (OpenCV)
├── .streamlit/config.toml  Colour theme and upload limit
├── src/
│   ├── config.py           1. Folders and constants
│   ├── data.py             2. Image loading, dataset class, ZIP extraction
│   ├── models.py           3. Model zoo, loading, prediction, confidence (the AI core)
│   ├── database.py         4. SQLite history
│   ├── training.py         5. Transfer-learning training loop
│   ├── evaluation.py       6. Metrics and confusion matrix
│   ├── overlay.py          7. On-screen prediction panel for the camera
│   └── ui.py               Theme (CSS) and reusable components
└── views/                  One file per page
    ├── home.py  classify.py  dashboard.py  history.py
    └── evaluate.py  train.py  camera.py  compare.py
```

### How the AI part works (useful for your report or viva)

1. **CNN** – a Convolutional Neural Network learns visual features (edges, textures, shapes) in stacked layers, then a final layer scores every category.
2. **Softmax and confidence** – the scores are turned into probabilities that add up to 100%. The highest one is the **prediction confidence**. Below 50% the app warns that the model is unsure.
3. **Pretrained models** – MobileNetV2, ResNet18, ResNet50 and EfficientNet-B0 were trained on ImageNet (1.3 million images, 1,000 categories).
4. **Transfer learning** – for your own categories the app keeps the pretrained feature layers and replaces the final layer with one that has your number of categories. Only the new layer (or all layers, if you untick the option) is trained. That is why small datasets work.
5. **Data augmentation** – during training, images are randomly cropped, flipped and colour-shifted so the model generalises better.
6. **Evaluation metrics**
   - *Accuracy*: share of images classified correctly.
   - *Precision*: of the images the model labelled X, how many really were X.
   - *Recall*: of all real X images, how many the model found.
   - *F1*: the balance of precision and recall.
   - *Confusion matrix*: shows exactly which categories get mixed up.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the virtual environment (Step 3), then run `pip install -r requirements.txt` again. |
| First prediction is slow | Pretrained weights are downloading once. Later runs are instant. |
| Live stream cannot open the camera | Close Zoom/Teams/browser tabs using the camera, or change the camera number to 1. Live streaming works when the app runs on the same computer as the camera. |
| Evaluate says folder names do not match | Custom models need test folders named exactly like the training categories. ImageNet models need ImageNet names (`goldfish`, `tabby`). The easiest path is to train a custom model on the sample dataset and evaluate it on its test folder. |
| Training is slow | Use MobileNetV2, keep *only train the final layer* ticked and use 3 to 5 epochs. A GPU speeds it up a lot. |
| Out of memory while training | Lower the batch size to 8 or 16. |

## Ideas to extend

- Export the model to ONNX and serve it with FastAPI.
- Add Grad-CAM heatmaps to show which part of the image drove the prediction.
- Add a login page and per-user history.
