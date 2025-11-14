#!/usr/bin/env python3
"""
Make a 40 Hz pupil trace using Neo's FIR downsample, aligned to neural timebase.

- Reads eye *.nix* that contains 30 kHz AnalogSignals: XPos, YPos, XDiam, YDiam.
- Downsamples those signals with sig.downsample(factor=750, ftype='fir') → 40 Hz.
- Scales diameters (/1000), optional clamp of negatives to 0 (off by default).
- Computes pupil magnitude at 40 Hz.
- Aligns to neural session by intersecting millisecond timestamps at 25 ms step.
- Saves time_ms_25 (absolute ms) and pupil_40hz (float32) to .npz.

Usage:
  python export_pupil_40hz.py --eye-nix EYE.nix --neural-nix NEURAL.nix --out OUT.npz [--clamp-negatives]
"""

import argparse, json
import numpy as np
from pathlib import Path
from neo.io import NixIO

def _ms_time_from_anasig(sig):
    fs = float(sig.sampling_rate)      # Hz
    t0 = float(sig.t_start)            # seconds
    N  = int(np.array(sig).shape[0])
    t_ms = np.round((t0 + np.arange(N, dtype=np.float64)/fs) * 1000.0).astype(np.int64)
    return t_ms, fs, t0, N

def export_pupil_40hz(
    eye_nix_path: str,
    neural_nix_path: str,
    out_npz_path: str,
    clamp_negatives: bool = False,
):
    # ---------- load eye NIX (30 kHz) ----------
    with NixIO(eye_nix_path, "ro") as io:
        eblk = io.read_block()
    eseg = eblk.segments[0]
    xdiam_30k = eseg.analogsignals[2]   # XDiam @ 30 kHz
    ydiam_30k = eseg.analogsignals[3]   # YDiam @ 30 kHz

    fs_eye_30k = float(xdiam_30k.sampling_rate)   # should be 30000.0
    # sanity: expect 30000
    if int(round(fs_eye_30k)) != 30000:
        print(f"[warn] eye sampling_rate is {fs_eye_30k} Hz (expected 30000). Proceeding anyway.")

    # ---------- downsample each component using authors' method (FIR) ----------
    factor = 750  # 30000 / 750 = 40 Hz
    x40 = xdiam_30k.downsample(factor, ftype="fir")  # AnalogSignal at 40 Hz
    y40 = ydiam_30k.downsample(factor, ftype="fir")

    # ---------- scale & (optional) clamp (post-downsample, like authors’ stage) ----------
    # authors divided by 1000 at the downsampled stage
    x = (np.array(x40).squeeze().astype(np.float64) / 1000.0)
    y = (np.array(y40).squeeze().astype(np.float64) / 1000.0)

    if clamp_negatives:
        # matching their spirit, but at 40 Hz (much less destructive than at 1 kHz / 30 kHz)
        x[x < 0] = 0.0
        y[y < 0] = 0.0

    pupil_40 = np.sqrt(x*x + y*y).astype(np.float32)

    # eye 40 Hz time (absolute ms since recording start)
    time_ms_eye_40, fs_eye_40, t0_eye, N_eye_40 = _ms_time_from_anasig(x40)  # fs_eye_40 ≈ 40

    # ---------- load neural to define overlap on the same ms origin ----------
    with NixIO(neural_nix_path, "ro") as io:
        nblk = io.read_block()
    nsig_1k = nblk.segments[0].analogsignals[0]
    time_ms_neu_1k, fs_neu, t0_neu, N_neu = _ms_time_from_anasig(nsig_1k)   # fs_neu ≈ 1000

    # Build the neural 25 ms grid (40 Hz) in absolute ms
    # Start from the first 25 ms boundary >= neural t_start, end before neural t_stop
    neu_start_ms = int(time_ms_neu_1k[0])
    neu_end_ms   = int(time_ms_neu_1k[-1])
    # align start to a 25 ms boundary
    start_25 = neu_start_ms + ((25 - (neu_start_ms % 25)) % 25)
    time_ms_neu_25 = np.arange(start_25, neu_end_ms + 1, 25, dtype=np.int64)

    # ---------- intersect eye 40 Hz ms grid with neural 25 ms grid ----------
    # Both are integer ms; intersection makes sure we keep only overlap and exact timepoints.
    common_ms, idx_neu, idx_eye = np.intersect1d(time_ms_neu_25, time_ms_eye_40, assume_unique=False, return_indices=True)

    pupil_40_aligned = pupil_40[idx_eye]  # aligned to common_ms

    # ---------- save ----------
    meta = dict(
        source_eye=str(eye_nix_path),
        source_neural=str(neural_nix_path),
        fs_eye_native=fs_eye_30k,
        fs_eye_downsampled=fs_eye_40,
        fs_neural=fs_neu,
        t0_eye_s=t0_eye,
        t0_neural_s=t0_neu,
        n_eye_40=int(N_eye_40),
        n_neural=int(N_neu),
        n_out=int(common_ms.size),
        downsample={"factor": factor, "ftype": "fir"},
        processing=f"/1000 on diameters; clamp_negatives={clamp_negatives}; pupil=sqrt(x^2+y^2) at 40 Hz",
    )

    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz_path,
        time_ms_25=common_ms,             # absolute ms (25 ms steps)
        pupil_40hz=pupil_40_aligned,      # float32
        meta=json.dumps(meta),
    )

    print(f"Saved 40 Hz pupil to: {out_npz_path}")
    print(f"  samples: {common_ms.size} (25 ms grid, aligned to neural)")
    print(f"  first/last time_ms_25: {common_ms[0]} .. {common_ms[-1]}")
    print(f"  example values (first 5): {pupil_40_aligned[:5]}")
    return common_ms, pupil_40_aligned, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-nix", required=True, help="Path to *_aligned_eye_data.nix")
    ap.add_argument("--neural-nix", required=True, help="Path to NSP*_array*_MUAe.nix (1 kHz)")
    ap.add_argument("--out", help="Output .npz path (default: derive from eye file name)")
    ap.add_argument("--clamp-negatives", action="store_true", help="Clamp negatives to 0 after downsampling (off by default)")
    args = ap.parse_args()

    eye = Path(args.eye_nix)
    neu = Path(args.neural_nix)
    outp = Path(args.out) if args.out else eye.with_name(eye.stem.replace("_aligned_eye_data", "") + "_pupil_40hz_25ms.npz")

    export_pupil_40hz(str(eye), str(neu), str(outp), clamp_negatives=args.clamp_negatives)

if __name__ == "__main__":
    main()
