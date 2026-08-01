# RF Signal Classification using Deep Learning

## Project Overview

The goal of this project is to build a system that can listen to the radio frequency spectrum in real time and automatically identify what type of signal it is hearing,without any human intervention. Given a raw stream of IQ samples from a software-defined radio dongle, the system should classify the signal into one of several known categories with high confidence.

This sits at the intersection of signal processing and deep learning. Rather than hand-crafting features from the RF spectrum, we let convolutional neural networks learn directly from the data, either from the raw time-domain waveform or from a visual time-frequency representation of it.

The system supports both offline inference on pre-recorded `.npy` captures and live classification using an SDR.

The project includes two CNN-based approaches:

- **1D SignalCNN**: learns directly from raw IQ waveforms.
- **2D SpectrogramCNN**: converts IQ windows into spectrograms and learns time-frequency patterns.

---
## Project Workflow

The overall workflow of the project is illustrated below:

<img width="1000" height="1100" alt="pipeline drawio" src="https://github.com/user-attachments/assets/9482acdc-76c7-4a50-9202-938c5509d560" />

## What the System Does

The overall pipeline is:

1. An RTL-SDR captures raw IQ samples at a selected center frequency.
2. The IQ samples are normalized and divided into windows.
3. The samples are converted into the representation required by the selected model.
4. A CNN predicts the signal class.
5. The prediction is returned with a confidence score.

The models can also be used with previously captured `.npy` files, so an SDR is not required for offline inference.

### Signal Classes

| Class | Description |
|---|---|
| ADS_B | Aircraft transponder signals around 1090 MHz |
| FM_broadcast | Commercial FM radio signals |
| ISM_sensors | Signals from ISM-band devices, including 433 MHz systems |
| noise | Background RF noise / no target signal |

<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/49c8093d-0f62-449f-bcc0-981943875679" />

---

## Dataset

The dataset utlized was the [TrevTron/rtl-ml-dataset](https://huggingface.co/datasets/TrevTron/rtl-ml-dataset) along with simulations generated on Simulink and signals collected using an RTL-SDR dongle across multiple capture sessions. Each recording is stored as a `.npy` file containing a dict with the raw complex IQ samples and capture metadata (center frequency, sample rate, timestamp, label).

```
data/datasets_validated/
├── ADS_B/          — 160 recordings
├── FM_broadcast/   — 160 recordings  (multiple FM stations: 91.1, 93.5, 95.0, 98.3, 104.8, 106.4 MHz)
├── ISM_sensors/    — 160 recordings
└── noise/          — 160 recordings  (captured across 50, 300, 470, 800, 1200 MHz)
```

All recordings are split stratified per class: **70% train / 10% validation / 20% test**. The split is done at the recording level so windows from the same capture never appear in both train and test.

---

## Models

### 1D Model: SignalCNN

The 1D model operates directly on the raw IQ signal.

Each 2048-sample window is represented using two channels:

- I: in-phase component
- Q: quadrature component

Input shape:

```text
[B, 2, 2048]
```

The model uses a series of `Conv1d` blocks with BatchNorm, ReLU, and MaxPool layers, followed by adaptive average pooling, dropout, and a linear classifier.

### 2D Model: SpectrogramCNN

The 2D model converts each 2048-sample IQ window into a `128 × 128` spectrogram.

Input shape:

```text
[B, 1, 128, 128]
```

The spectrogram is generated using an STFT with:

- 256-sample Hann window
- 50% overlap
- log compression
- min-max normalization
- resizing to `128 × 128`

The CNN uses multiple `Conv2d` blocks with BatchNorm, ReLU, and MaxPool layers, followed by adaptive average pooling, dropout, and a linear classifier.

Spectrograms are generated during data loading rather than storing the entire spectrogram dataset as `.npy` files. This keeps disk and RAM usage much lower.

---

## Spectrograms

A spectrogram represents the signal in the time-frequency domain:

- **X-axis:** time
- **Y-axis:** frequency
- **Brightness:** signal magnitude/power

Typical signal patterns include:

- **ADS-B:** short bursts
- **FM broadcast:** continuous wide-band structure
- **ISM sensors:** narrow/intermittent transmissions
- **Noise:** diffuse and less structured RF energy

Example spectrograms can be generated locally using the project's spectrogram utilities.

| ADS_B | FM_broadcast | ISM_sensors | noise |
|---|---|---|---|
| <img width="250" height="250" alt="ADS-B spectrogram" src="https://github.com/user-attachments/assets/438751bb-379e-4557-96e2-e8822e48b650" /> | <img width="250" height="250" alt="FM broadcast spectrogram" src="https://github.com/user-attachments/assets/b015c71b-bcd9-426b-a83b-9bbe73f89d82" /> | <img width="250" height="250" alt="ISM sensors spectrogram" src="https://github.com/user-attachments/assets/947e8a69-5c81-4a8e-973a-f1413128b4ba" /> | <img width="250" height="250" alt="Noise spectrogram" src="https://github.com/user-attachments/assets/be0fd365-ad79-491d-adc6-2e21131d5fa7" /> |


---

## SDR Hardware

The SDR used for this project was:

**Nooelec NESDR SMArt v5 SDR - HF/VHF/UHF (100 kHz–1.75 GHz) RTL-SDR, RTL2832U & R820T2-Based Software Defined Radio**

The SDR is used to capture raw IQ samples from the RF environment.

The capture pipeline supports:

- configurable center frequency
- configurable sample count
- configurable sample rate
- automatic gain
- saving captures as `.npy` files

The default sample rate used in the project is:

```text
1,024,000 samples/second
```

## SDR Interaction

The RTL-SDR dongle is the hardware interface between the physical RF environment and the software pipeline. It is a low-cost USB receiver that can tune across roughly 24 MHz to 1.7 GHz and stream raw IQ samples to the host machine.

`live_capture.py` handles all SDR interaction:
- Configures center frequency, sample rate (default 1,024,000 Hz), and gain
- Reads IQ samples in chunks to avoid USB buffer overflows
- Wraps captures in a metadata dict and saves as `.npy` — the exact same format as the training dataset

> <img width="959" height="503" alt="Screenshot 2026-07-28 143843" src="https://github.com/user-attachments/assets/12c56b21-0c7c-4969-a57d-ba7705ddb5c3" />
> <img width="959" height="502" alt="Screenshot 2026-07-29 105957" src="https://github.com/user-attachments/assets/58781b4a-b216-494b-b725-dd9a10b99c61" />

---

## Using Your Own Captured Samples

You can use your own SDR recordings to test the trained models.

The project also includes a `live_capture.py` script that allows users with a compatible RTL-SDR to capture raw IQ samples directly from the RF environment.

The workflow is:

- Connect the RTL-SDR to the system.
- Specify the center frequency and number of samples to capture.
- Capture raw IQ samples from the selected frequency.
- Save the capture as a `.npy` file for later use.
- Create a `predict_samples` directory in your local project setup
- Place the captured file in the `predict_samples/` directory to run offline inference using either the 1D or 2D model.
- Alternatively, use the live classifier to perform classification directly from the SDR.

For example:

```bash
python src/live_capture.py 1090000000 --output predict_samples/live.npy
```

---

## Live Classification

If you have a compatible RTL-SDR, you can also use the live classifier.

The live workflow is:

```text
SDR → IQ samples → preprocessing → trained CNN → predicted class + confidence
```

The live classifier requires:

- a connected RTL-SDR
- a center frequency
- a trained model checkpoint

For example, the classifier can be run by providing the center frequency and model to the live-classification script.

Refer to the command-line arguments in the [Run live SDR classification](#8-run-live-sdr-classification) section below.

---

## Results

### 1D SignalCNN

The 1D model was evaluated on the four-class test set.

| Metric | Score |
|---|---:|
| Test Accuracy | **99.95%** |
| Macro Precision | **99.95%** |
| Macro Recall | **99.95%** |
| Macro F1 | **99.95%** |
| Test Loss | **0.0029** |

```text
              precision    recall  f1-score   support

       ADS_B       1.00      1.00      1.00      2111
FM_broadcast       1.00      1.00      1.00      2000
 ISM_sensors       1.00      1.00      1.00      2298
       noise       1.00      1.00      1.00      1848

    accuracy                           1.00      8257
```

> <img width="2717" height="2365" alt="image" src="https://github.com/user-attachments/assets/0d1b8d02-338c-43af-aab2-1c083fc55649" />
> <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/fa47a122-9af4-44f6-9461-7c5acba5351b" />
> <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/839ab70e-d894-4116-9fa9-7feff4715e37" />

### 2D SpectrogramCNN

The 2D model was evaluated on 31,443 test windows.

| Metric | Score |
|---|---:|
| Test Accuracy | **93.2%** |
| Precision | **93.7%** |
| Recall | **93.1%** |
| F1-score | **93.1%** |
| Test Loss | **0.16** |

```text
              precision    recall  f1-score   support

       ADS_B       0.83      0.95      0.89      8000
FM_broadcast       0.98      1.00      0.99      7598
 ISM_sensors       0.99      1.00      0.99      8149
       noise       0.95      0.78      0.86      7696

    accuracy                           0.93     31443
   macro avg       0.94      0.93      0.93     31443
weighted avg       0.94      0.93      0.93     31443
```

> <img width="2717" height="2365" alt="image" src="https://github.com/user-attachments/assets/d7649412-1e5c-4355-844b-099b63cfa38d" />
> <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/3e34b262-8f69-4ea5-8705-d9f401e72ed0" />
> <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/2a7658b3-6e03-436d-b34a-582725358cbe" />

### Model Comparison

| | 1D SignalCNN | 2D SpectrogramCNN |
|---|---|---|
| Representation | Raw IQ waveform | STFT spectrogram |
| Input | `[B, 2, 2048]` | `[B, 1, 128, 128]` |
| Test Accuracy | **99.95%** | **93.2%** |
| Macro F1 | **99.95%** | **93.1%** |
| Test Loss | **0.0029** | **0.16** |
| Main advantage | High classification accuracy | Time-frequency representation |
| Main trade-off | Raw waveform representation | Higher preprocessing cost |

The 1D model achieved higher test accuracy on its four-class problem. The 2D model provides a time-frequency representation that makes signal structure easier to visualize and interpret.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Deep learning | PyTorch |
| Signal processing | SciPy |
| Numerical computing | NumPy |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| SDR interface | pyrtlsdr |
| Notebooks | Jupyter / ipykernel |
| Progress bars | tqdm |
| Python | 3.11+ |
| SDR | Nooelec NESDR SMArt v5 |

---

## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/Ishita-190/RF-Signal-Classification-using-Deep-Learning.git
cd RF-Signal-Classification-using-Deep-Learning
```

### 2. Create a virtual environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required dataset files in the expected local dataset directory.

The repository intentionally does not include the full dataset or trained model checkpoints.

### 5. Train the 1D model

The project includes two Jupyter notebooks, one for each model: **experiements_1d.ipynb** and **experiements_2d.ipynb**. Each notebook runs the complete pipeline from the raw dataset to model evaluation.

The overall process in each notebook is:

1. **Load the dataset** and map the RF recordings to their corresponding signal classes.
2. **Preprocess the raw IQ samples** according to the requirements of the selected model.
3. **Split the dataset** into stratified training, validation, and test sets.
4. **Prepare the model inputs**:

   * 1D model: normalized I/Q windows.
   * 2D model: 128×128 spectrograms generated from the IQ windows using STFT.
5. **Initialize and train the CNN**, while tracking training and validation loss and accuracy.
6. **Evaluate the trained model** on the unseen test set.
7. **Calculate evaluation metrics**, including accuracy, precision, recall, F1-score, and test loss.
8. **Generate visualizations and results**, such as confusion matrices, training curves, and example spectrograms for the 2D model.

Users can simply run the cells in the respective notebook in order to reproduce the complete pipeline without having to manually execute each individual preprocessing, training, or evaluation script.


### 7. Run offline prediction

Place your captured `.npy` files in:

```text
predict_samples/
```

Then run the appropriate prediction module:

```bash
cd src
python -m model_1d.predict
```

or:

```bash
cd src
python -m model_2d.predict_2d
```

### 8. Run live SDR classification

Connect the Nooelec NESDR SMArt v5 or another compatible RTL-SDR and provide the required center frequency and trained model to the live classifier.

To capture a sample and run inference:

```bash
# Capture 512k samples at 1090 MHz (ADS-B)
python src/live_capture.py 1090000000 --num-samples 512000 --output predict_samples/live.npy

# Run 1D model inference
cd src && python -m model_1d.predict

# Run 2D model inference
cd src && python -m model_2d.predict_2d
```
Or, you can connect the SDR to your laptop, tune it to the required frequency using SDR++ and run the following commands

```bash
# run the 1D CNN model for any of the 4 signal types (ex - noise)
python live_classifier.py 920000000 --model 1d

# run the 2D CNN model for any of the 4 signal types (ex - FM broadcast)
python live_classifier.py 93500000 --model 2d
```
You will obtain live predictions in this manner:

><img width="577" height="365" alt="Screenshot 2026-07-31 140127" src="https://github.com/user-attachments/assets/d2f9723c-92a4-41e6-9423-369877b8bb43" />

><img width="634" height="317" alt="Screenshot 2026-07-31 135812" src="https://github.com/user-attachments/assets/d65b026d-de70-4e18-9430-cdc29026c146" />

---

