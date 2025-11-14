#!/usr/bin/env python3
"""
Export pupil trace (1 ms) aligned to neural time base using true t_start.

- Eye NIX is expected to contain 30 kHz AnalogSignals: XPos, YPos, XDiam, YDiam (in mV).
- Neural NIX is expected to contain a 1 kHz AnalogSignal (your MUA/LFP time grid).

We:
  1) Build the neural 1 kHz time grid from its t_start and length.
  2) Build the eye 30 kHz time grid from its t_start and length.
  3) Interpolate XDiam/YDiam -> neural grid; clamp negatives to 0 and divide by 1000 (authors' logic).
  4) Compute pupil = sqrt(XDiam^2 + YDiam^2).
  5) Trim to the exact temporal overlap and save:
       - time_ms : absolute ms since recording start (rounded)
       - pupil   : float32, aligned to neural grid, trimmed to overlap
       - meta    : JSON (string) with provenance
"""

import argparse
import json
import numpy as np
from pathlib import Path
from neo.io import NixIO


def export_pupil_aligned(eye_nix_path: Path, neural_nix_path: Path, out_npz_path: Path):
    # --- load eye (30 kHz) ---
    with NixIO(str(eye_nix_path), 'ro') as io:
        eye_blk = io.read_block()
    eye_seg = eye_blk.segments[0]

    # Expect order: 0 XPos, 1 YPos, 2 XDiam, 3 YDiam (as you printed)
    xdiam_sig = eye_seg.analogsignals[2]
    ydiam_sig = eye_seg.analogsignals[3]

    XDiam = np.array(xdiam_sig).squeeze().astype(np.float64)
    YDiam = np.array(ydiam_sig).squeeze().astype(np.float64)

    fs_eye = float(xdiam_sig.sampling_rate)   # should be 30000.0
    t0_eye = float(xdiam_sig.t_start)         # seconds (likely 0.0)

    # Authors' preprocessing: divide diam by 1000 and clamp negatives to 0
    XDiam = (XDiam / 1000.0)
    YDiam = (YDiam / 1000.0)

    N_eye = XDiam.size
    t_eye = t0_eye + np.arange(N_eye, dtype=np.float64) / fs_eye

    # --- load neural (1 kHz) and build authoritative time grid ---
    with NixIO(str(neural_nix_path), 'ro') as io:
        neu_blk = io.read_block()
    neu_sig = neu_blk.segments[0].analogsignals[0]

    fs_neu = float(neu_sig.sampling_rate)     # expect 1000.0
    t0_neu = float(neu_sig.t_start)           # seconds
    N_neu  = int(np.array(neu_sig).shape[0])
    t_neu  = t0_neu + np.arange(N_neu, dtype=np.float64) / fs_neu  # seconds

    # --- align by true starts: interpolate onto neural grid, trim to overlap ---
    tmin = max(t_eye[0], t_neu[0])
    tmax = min(t_eye[-1], t_neu[-1])
    if tmax <= tmin:
        raise ValueError("No temporal overlap between eye and neural time ranges.")

    in_overlap = (t_neu >= tmin) & (t_neu <= tmax)

    x1k = np.full(N_neu, np.nan, dtype=np.float32)
    y1k = np.full(N_neu, np.nan, dtype=np.float32)
    x1k[in_overlap] = np.interp(t_neu[in_overlap], t_eye, XDiam).astype(np.float32)
    y1k[in_overlap] = np.interp(t_neu[in_overlap], t_eye, YDiam).astype(np.float32)

    pupil_on_neu = np.sqrt(x1k**2 + y1k**2).astype(np.float32)
    # Trim to overlap so there are no NaNs
    i0 = int(np.argmax(in_overlap))
    i1 = int(in_overlap.size - np.argmax(in_overlap[::-1]))  # slice end (exclusive)

    t_neu_trim = t_neu[i0:i1]                    # seconds
    pupil_trim = pupil_on_neu[i0:i1]             # float32

    # Convert to absolute ms since recording start (using actual neural t_start)
    time_ms = np.round(t_neu_trim * 1000.0).astype(np.int64)


    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(
        source_eye=str(eye_nix_path),
        source_neural=str(neural_nix_path),
        fs_eye=fs_eye,
        fs_neural=fs_neu,
        t0_eye=t0_eye,
        t0_neural=t0_neu,
        N_eye=N_eye,
        N_neural=N_neu,
        trimmed_start_ms=int(time_ms[0]),
        trimmed_end_ms=int(time_ms[-1]),
        method="interp_eye_to_neural_grid_then_trim_overlap",
        pupil_units="arbitrary (diam scaled /1000, combined as sqrt(x^2+y^2) after interp; no per-sample clamping)"
    )

    # Save with JSON metadata (string) to avoid np.savez dict errors
    np.savez_compressed(
        out_npz_path,
        time_ms=time_ms,
        pupil=pupil_trim.astype(np.float32),
        meta=json.dumps(meta)
    )

    print(f"Saved pupil to: {out_npz_path}")
    print(f"  length: {len(pupil_trim)} samples (should match length of trimmed neural window)")
    print(f"  first/last time_ms: {time_ms[0]} .. {time_ms[-1]}")


def main():
    ap = argparse.ArgumentParser(description="Export aligned 1 ms pupil trace from eye+neural NIX.")
    ap.add_argument("--eye-nix", required=True, help="Path to *_aligned_eye_data.nix")
    ap.add_argument("--neural-nix", required=True, help="Path to NSP*_array*_MUAe.nix (or your 1 kHz neural NIX)")
    ap.add_argument("--out", help="Output .npz path (default: derive from eye file name)")
    args = ap.parse_args()

    eye = Path(args.eye_nix)
    neu = Path(args.neural_nix)
    if args.out:
        outp = Path(args.out)
    else:
        # e.g., L_RS_090817_pupil_1ms.npz
        stem = eye.stem.replace("_aligned_eye_data", "") + "_pupil_1ms.npz"
        outp = eye.with_name(stem)

    export_pupil_aligned(eye, neu, outp)


if __name__ == "__main__":
    main()
