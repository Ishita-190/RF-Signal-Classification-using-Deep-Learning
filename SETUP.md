# Setup Guide — Running from Scratch

## 1. Clone the repository

```bash
git clone https://github.com/Ishita-190/RF-Signal-Classification-using-Deep-Learning.git
cd RF-Signal-Classification-using-Deep-Learning
```

---

## 2. Create and activate a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download the dataset

The dataset will be downloaded automatically when you run the notebooks.

Alternatively, download it manually:

```bash
cd src
python -c "from dataset import download_dataset; download_dataset()"
cd ..
```

This downloads ~3GB from [ishisan28/rf-signal-dataset](https://huggingface.co/datasets/ishisan28/rf-signal-dataset) into `data/datasets_validated/`.

---

## 5. Train and evaluate

Open and run all cells in the notebooks in order:

- `notebooks/experiments_1d.ipynb` — 1D SignalCNN
- `notebooks/experiments_2d.ipynb` — 2D SpectrogramCNN

Each notebook will:
1. Download the dataset (if not already present)
2. Preprocess the IQ samples
3. Train the CNN
4. Evaluate on the test set
5. Save the model to `models/best_model_1d.pth` or `models/best_model_2d.pth`
6. Save results to `results/`

---

## 6. Run offline prediction

Place your `.npy` capture files in `predict_samples/`, then:

```bash
# 1D model
cd src
python -m model_1d.predict

# 2D model
cd src
python -m model_2d.predict_2d
```

---

## 7. Capture live IQ samples (requires RTL-SDR)

```bash
python src/live_capture.py 1090000000 --num-samples 512000 --output predict_samples/live.npy
```

---

## 8. Run live classification (requires RTL-SDR)

```bash
# 1D model
python src/live_classifier.py 920000000 --model 1d

# 2D model
python src/live_classifier.py 93500000 --model 2d
```

---

## Notes

- Python 3.11+ required
- GPU is optional — the models will train on CPU but will be slower for the 2D model
- If you already have the dataset, the download step will be skipped automatically
- Pre-trained model checkpoints are included in the repository (`models/best_model_1d.pth`, `models/best_model_2d.pth`) — you can skip training and go straight to prediction if needed
