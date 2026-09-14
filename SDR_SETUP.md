# SDR Setup Guide

This guide covers installing Zadig, SDR++, and configuring the RTL-SDR dongle for use with this project on Windows.

---

## 1. Install Zadig (USB Driver)

Zadig replaces the default Windows USB driver for the RTL-SDR dongle with WinUSB, which is required for `pyrtlsdr` to communicate with the hardware.

1. Download Zadig from https://zadig.akeo.ie
2. Plug in the RTL-SDR dongle
3. Open Zadig
4. Click **Options → List All Devices**
5. In the dropdown, select **Bulk-In, Interface (Interface 0)** — this is the RTL-SDR device
6. In the driver box on the right, select **WinUSB**
7. Click **Replace Driver** and wait for it to complete
8. Close Zadig

> If you later need to use the dongle with other software that requires the original driver, you can revert via Device Manager → right-click the device → Update driver.

---

## 2. Install SDR++

SDR++ is used to visually tune and monitor the RF spectrum before or during capture sessions.

1. Download the latest SDR++ Windows release from https://github.com/AlexandreRouma/SDRPlusPlus/releases
2. Extract the zip to a folder of your choice (e.g. `C:\SDRPlusPlus\`)
3. Inside the extracted folder, locate the following DLL files:
   - `libusb-1.0.dll`
   - `pthreadVC2.dll`
   - `rtlsdr.dll`
4. Copy all three DLL files into your project's virtual environment Scripts directory:
   ```
   .venv\Scripts\
   ```
   These are required by `pyrtlsdr` at runtime to interface with the dongle.
5. Run `sdrpp.exe` from the extracted SDR++ folder to launch the application

---

## 3. Configure SDR++ for RTL-SDR

1. Open SDR++
2. In the left panel under **Source**, select **RTL-SDR** from the dropdown
3. Click the **Play** button (▶) at the top to start the stream
4. Use the frequency bar at the top to tune to your target frequency:
   - ADS-B: `1090.000 MHz`
   - FM broadcast: e.g. `93.500 MHz`, `91.100 MHz`, `95.000 MHz`
   - ISM sensors: `433.920 MHz` or `915.000 MHz`
   - Noise: any quiet frequency e.g. `300.000 MHz`
5. You should see a live waterfall and spectrum display confirming the dongle is working

---

## 4. Verify the Setup in Python

With the dongle plugged in and the DLLs copied, verify `pyrtlsdr` can detect the device:

```python
from rtlsdr import RtlSdr
sdr = RtlSdr()
print("Device open:", sdr)
sdr.close()
```

If no error is raised, the driver and DLLs are correctly configured.

---

## 5. Capture Samples

```bash
# Capture 512k samples at 1090 MHz (ADS-B)
python src/live_capture.py 1090000000 --num-samples 512000 --output predict_samples/live.npy
```

See the main README for full capture and inference instructions.
