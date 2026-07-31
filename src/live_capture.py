from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rtlsdr import RtlSdr


DEFAULT_SAMPLE_RATE = 1_024_000
DEFAULT_NUM_SAMPLES = 512_000
DEFAULT_CHUNK_SIZE = 256 * 1024
DEFAULT_GAIN = "auto"


def configure_sdr(
    sdr: RtlSdr,
    center_freq: float,
    sample_rate: float,
    gain: str | int,
) -> None:
    sdr.center_freq = center_freq
    sdr.sample_rate = sample_rate
    sdr.gain = gain


def read_iq_samples(
    sdr: RtlSdr,
    num_samples: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    samples = []
    remaining = num_samples

    while remaining > 0:
        count = min(chunk_size, remaining)
        samples.append(sdr.read_samples(count))
        remaining -= count

    return np.asarray(np.concatenate(samples), dtype=np.complex64)


def main():
    parser = argparse.ArgumentParser(description="Capture IQ samples using an RTL-SDR.")

    parser.add_argument("center_freq", type=float, help="Center frequency (Hz)")
    parser.add_argument("--output", type=str, help="Output .npy file")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--gain", default=DEFAULT_GAIN)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--label", default=None)

    args = parser.parse_args()

    gain = int(args.gain) if str(args.gain).isdigit() else args.gain

    sdr = RtlSdr(device_index=args.device_index)

    try:
        configure_sdr(
            sdr,
            center_freq=args.center_freq,
            sample_rate=args.sample_rate,
            gain=gain,
        )

        samples = read_iq_samples(sdr, args.num_samples)

    finally:
        sdr.close()

    record = {
        "samples": samples,
        "center_freq": float(args.center_freq),
        "sample_rate": float(args.sample_rate),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "label": args.label,
        "duration": len(samples) / args.sample_rate,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, record, allow_pickle=True)
        print(f"Saved to {output.resolve()}")

    print(f"Captured {len(samples)} IQ samples")
    print(f"Center Frequency: {args.center_freq:.0f} Hz")
    print(f"Sample Rate: {args.sample_rate:.0f} Hz")
    print(f"Duration: {record['duration']:.3f} s")


if __name__ == "__main__":
    main()
