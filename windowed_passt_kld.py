#!/usr/bin/env python3
"""
Windowed PaSST KLD evaluation for structural collapse.
v2: robust error handling, correct hear21passt API, explicit failure reporting.

Changes vs v1:
  - Errors are NOT silently swallowed. Each failure is logged with a reason.
  - PaSST call uses the documented hear21passt API: model(wav) where wav is
    (B, T) at 32 kHz, and the output is logits (B, 527).
  - Adds --smoke_test flag to just run the first clip through end-to-end and
    print everything, so you can see if the pipeline works at all.

Usage:
    # Smoke test first (one clip, verbose):
    python windowed_passt_kld_v2.py \
        --metadata /mnt/user-data/uploads/MusicBench_test_A.json \
        --reference_root /storage/ssd1/richtsai1103/MusicBench/datashare \
        --gen_root /storage/ssd3/richtsai1103/MusicBench \
        --models MusicGen_SASS MusicGen_TF \
        --smoke_test

    # Full run:
    python windowed_passt_kld_v2.py \
        --metadata /mnt/user-data/uploads/MusicBench_test_A.json \
        --reference_root /storage/ssd1/richtsai1103/MusicBench/datashare \
        --gen_root /storage/ssd3/richtsai1103/MusicBench \
        --models MusicGen_SASS MusicGen_TF \
        --out_dir ./windowed_kld_results
"""

import argparse
import json
import os
import traceback
import typing as tp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


PASST_SR = 32000
PASST_WINDOW_SEC = 10.0
PASST_TARGET_LEN = int(PASST_SR * PASST_WINDOW_SEC)  # 320000


# ---------------------------------------------------------------------------- #
# Compatibility shim: torch.stft in PyTorch >= 2.0 refuses to run without an    #
# explicit return_complex=True, but hear21passt / torchlibrosa call torch.stft  #
# without that argument. We patch the default here BEFORE importing PaSST so    #
# that every downstream stft call gets return_complex=True automatically.      #
# Symptom this fixes:                                                           #
#   RuntimeError: stft requires the return_complex parameter be given for real  #
#   inputs, and will further require that return_complex=True in a future      #
#   PyTorch release.                                                            #
# ---------------------------------------------------------------------------- #
def _patch_torch_stft():
    import functools
    _orig_stft = torch.stft

    @functools.wraps(_orig_stft)
    def _patched_stft(input, n_fft, hop_length=None, win_length=None,
                      window=None, center=True, pad_mode='reflect',
                      normalized=False, onesided=None, return_complex=None):
        if return_complex is None:
            return_complex = True
        out = _orig_stft(
            input, n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, pad_mode=pad_mode,
            normalized=normalized, onesided=onesided,
            return_complex=return_complex,
        )
        # torchlibrosa expects a real (..., 2) tensor, not a complex tensor.
        # Convert back so downstream code sees [real, imag] in the last dim.
        if torch.is_complex(out):
            out = torch.view_as_real(out)
        return out

    torch.stft = _patched_stft


_patch_torch_stft()


# ---------------------------------------------------------------------------- #
# PaSST loader — handles both get_basic_model and get_scene_embeddings APIs     #
# ---------------------------------------------------------------------------- #
def load_passt(device: str = "cuda"):
    """Load the PaSST AudioSet-tagging model."""
    from hear21passt.base import get_basic_model
    model = get_basic_model(mode="logits")
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def passt_forward(model, wav_batch: torch.Tensor, device: str) -> torch.Tensor:
    """
    wav_batch: (B, T) mono waveform at 32 kHz, exactly PASST_TARGET_LEN samples.
    Returns: (B, 527) softmax probabilities over AudioSet classes.

    The hear21passt get_basic_model expects (B, T) at 32 kHz. Some versions
    return just logits; others return (logits, embeddings). We handle both.
    """
    wav_batch = wav_batch.to(device)
    out = model(wav_batch)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    probs = torch.softmax(logits.float(), dim=-1).cpu()  # (B, 527)
    return probs


# ---------------------------------------------------------------------------- #
# Audio loading                                                                 #
# ---------------------------------------------------------------------------- #
def load_wav_for_passt(path: str) -> torch.Tensor:
    """Load, convert to mono, resample to 32 kHz, pad/truncate to 10s.
    Returns: (PASST_TARGET_LEN,) 1-D tensor."""
    wav, sr = torchaudio.load(path)  # (channels, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != PASST_SR:
        wav = torchaudio.functional.resample(wav, sr, PASST_SR)
    wav = wav.squeeze(0)  # (T,)
    if wav.shape[0] > PASST_TARGET_LEN:
        wav = wav[:PASST_TARGET_LEN]
    elif wav.shape[0] < PASST_TARGET_LEN:
        pad = PASST_TARGET_LEN - wav.shape[0]
        wav = F.pad(wav, (0, pad))
    return wav


# ---------------------------------------------------------------------------- #
# KLD                                                                           #
# ---------------------------------------------------------------------------- #
def kld_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p = p.double().clamp_min(eps)
    q = q.double().clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p.log() - q.log())).sum().item())


def compute_kld_pairwise(gen_probs: tp.List[torch.Tensor],
                         ref_probs: tp.List[torch.Tensor]) -> tp.Dict[str, float]:
    assert len(gen_probs) == len(ref_probs)
    fwd_list, rev_list, sym_list = [], [], []
    for p_gen, p_ref in zip(gen_probs, ref_probs):
        fwd = kld_pq(p_gen, p_ref)
        rev = kld_pq(p_ref, p_gen)
        fwd_list.append(fwd)
        rev_list.append(rev)
        sym_list.append(0.5 * (fwd + rev))
    return {
        "fwd_kld_mean": float(np.mean(fwd_list)),
        "fwd_kld_std":  float(np.std(fwd_list)),
        "rev_kld_mean": float(np.mean(rev_list)),
        "rev_kld_std":  float(np.std(rev_list)),
        "sym_kld_mean": float(np.mean(sym_list)),
        "sym_kld_std":  float(np.std(sym_list)),
        "n_pairs":      len(gen_probs),
    }


# ---------------------------------------------------------------------------- #
# Path builders                                                                 #
# ---------------------------------------------------------------------------- #
def ref_path_from_location(location: str, reference_root: str) -> str:
    return os.path.join(reference_root, location)


def gen_path_from_location(location: str, gen_root: str, model: str, seg: int) -> str:
    stem = Path(location).stem
    return os.path.join(
        gen_root, model, "generation_10s_split",
        f"seg{seg}", f"{stem}_seg{seg}.wav"
    )


# ---------------------------------------------------------------------------- #
# Metadata                                                                       #
# ---------------------------------------------------------------------------- #
def load_metadata(path: str) -> tp.List[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------- #
# Per-model PaSST prob extraction (verbose / batched)                           #
# ---------------------------------------------------------------------------- #
def compute_probs_for_model(
    model_name: str,
    metadata: tp.List[dict],
    reference_root: str,
    gen_root: str,
    passt_model,
    device: str,
    segments: tp.Sequence[int] = (0, 1, 2),
    batch_size: int = 16,
    verbose: bool = True,
    fail_log_path: tp.Optional[str] = None,
) -> tp.Dict[int, dict]:

    out = {s: {"gen_probs": [], "ref_probs": []} for s in segments}
    ref_cache: tp.Dict[str, torch.Tensor] = {}

    fail_counts = {
        "ref_load_err": 0,
        "gen_load_err": 0,
        "ref_missing": 0,
        "gen_missing": 0,
    }
    fail_log = []

    # Build the full list of (location, ref_path, [(seg, gen_path)...]) work items
    work_items = []
    for entry in metadata:
        location = entry["location"]
        ref_path = ref_path_from_location(location, reference_root)
        if not os.path.isfile(ref_path):
            fail_counts["ref_missing"] += 1
            fail_log.append(f"REF_MISSING: {ref_path}")
            continue
        seg_paths = []
        for seg in segments:
            gen_path = gen_path_from_location(location, gen_root, model_name, seg)
            if not os.path.isfile(gen_path):
                fail_counts["gen_missing"] += 1
                fail_log.append(f"GEN_MISSING: {gen_path}")
                continue
            seg_paths.append((seg, gen_path))
        if seg_paths:
            work_items.append((location, ref_path, seg_paths))

    if verbose:
        print(f"[{model_name}] {len(work_items)} clips have at least 1 valid segment.")

    # Process in batches for GPU efficiency. We batch across (clip, segment)
    # pairs for PaSST forward, then separately batch reference loads.
    iterator = tqdm(work_items, desc=f"[{model_name}]") if verbose else work_items

    for location, ref_path, seg_paths in iterator:
        # --- reference ---
        if ref_path in ref_cache:
            ref_p = ref_cache[ref_path]
        else:
            try:
                ref_wav = load_wav_for_passt(ref_path).unsqueeze(0)  # (1, T)
                ref_p = passt_forward(passt_model, ref_wav, device)[0]  # (527,)
                ref_cache[ref_path] = ref_p
            except Exception as e:
                fail_counts["ref_load_err"] += 1
                fail_log.append(f"REF_LOAD_ERR: {ref_path}: "
                                f"{type(e).__name__}: {e}")
                continue

        # --- gen segments (batched per clip) ---
        try:
            gen_wavs = [load_wav_for_passt(p) for _, p in seg_paths]
            gen_batch = torch.stack(gen_wavs, dim=0)  # (n_seg, T)
            gen_probs = passt_forward(passt_model, gen_batch, device)  # (n_seg, 527)
        except Exception as e:
            fail_counts["gen_load_err"] += 1
            fail_log.append(f"GEN_LOAD_ERR: {location}: "
                            f"{type(e).__name__}: {e}\n"
                            + traceback.format_exc())
            continue

        for i, (seg, _) in enumerate(seg_paths):
            out[seg]["gen_probs"].append(gen_probs[i])
            out[seg]["ref_probs"].append(ref_p)

    if verbose:
        print(f"[{model_name}] fail counts: {fail_counts}")
        for seg in segments:
            n = len(out[seg]["gen_probs"])
            print(f"  seg{seg}: {n} valid pairs")

    if fail_log_path and fail_log:
        with open(fail_log_path, "w") as f:
            for line in fail_log:
                f.write(line + "\n")
        if verbose:
            print(f"  Full failure log -> {fail_log_path}")

    return out


# ---------------------------------------------------------------------------- #
# Smoke test                                                                    #
# ---------------------------------------------------------------------------- #
def smoke_test(args, passt_model):
    """Run one clip end-to-end with maximum verbosity."""
    print("\n" + "=" * 70)
    print("SMOKE TEST — running first metadata entry end-to-end")
    print("=" * 70)
    metadata = load_metadata(args.metadata)
    entry = metadata[0]
    location = entry["location"]
    print(f"Using location: {location}")

    ref_path = ref_path_from_location(location, args.reference_root)
    print(f"\n1. Loading reference: {ref_path}")
    ref_wav = load_wav_for_passt(ref_path)
    print(f"   wav shape: {ref_wav.shape}, dtype: {ref_wav.dtype}, "
          f"min={ref_wav.min():.3f}, max={ref_wav.max():.3f}")

    print(f"\n2. Running PaSST forward on reference...")
    ref_p = passt_forward(passt_model, ref_wav.unsqueeze(0), args.device)[0]
    print(f"   probs shape: {ref_p.shape}, sum: {ref_p.sum():.4f}, "
          f"top5: {torch.topk(ref_p, 5).values.tolist()}")

    for model_name in args.models:
        print(f"\n3. Testing {model_name}:")
        for seg in args.segments:
            gen_path = gen_path_from_location(location, args.gen_root, model_name, seg)
            print(f"   seg{seg}: {gen_path}")
            gen_wav = load_wav_for_passt(gen_path)
            print(f"     wav shape: {gen_wav.shape}")
            gen_p = passt_forward(passt_model, gen_wav.unsqueeze(0), args.device)[0]
            fwd = kld_pq(gen_p, ref_p)
            rev = kld_pq(ref_p, gen_p)
            sym = 0.5 * (fwd + rev)
            print(f"     KL(gen||ref)={fwd:.4f}  KL(ref||gen)={rev:.4f}  sym={sym:.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — pipeline works end-to-end.")
    print("Run without --smoke_test for the full evaluation.")
    print("=" * 70)


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--reference_root", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--segments", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_dir", default="./windowed_kld_results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_clips", type=int, default=-1)
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading PaSST...")
    passt_model = load_passt(args.device)
    print("  PaSST loaded.")

    if args.smoke_test:
        smoke_test(args, passt_model)
        return

    print("Loading metadata...")
    metadata = load_metadata(args.metadata)
    if args.max_clips > 0:
        metadata = metadata[:args.max_clips]
    print(f"  {len(metadata)} clips.")

    all_results = {}

    for model_name in args.models:
        print(f"\n=== Processing {model_name} ===")
        fail_log_path = os.path.join(args.out_dir, f"{model_name}_failures.log")
        per_seg_probs = compute_probs_for_model(
            model_name=model_name,
            metadata=metadata,
            reference_root=args.reference_root,
            gen_root=args.gen_root,
            passt_model=passt_model,
            device=args.device,
            segments=args.segments,
            fail_log_path=fail_log_path,
        )

        model_results = {}
        for seg in args.segments:
            gen_probs = per_seg_probs[seg]["gen_probs"]
            ref_probs = per_seg_probs[seg]["ref_probs"]
            if len(gen_probs) == 0:
                print(f"  [WARN] seg{seg}: no valid pairs — check {fail_log_path}")
                continue
            pairwise = compute_kld_pairwise(gen_probs, ref_probs)
            seg_label = f"{seg*10}-{(seg+1)*10}s"
            model_results[seg_label] = pairwise
            print(f"  {seg_label}: "
                  f"sym_KL={pairwise['sym_kld_mean']:.4f}±{pairwise['sym_kld_std']:.4f}  "
                  f"fwd_KL={pairwise['fwd_kld_mean']:.4f}  "
                  f"rev_KL={pairwise['rev_kld_mean']:.4f}  "
                  f"(n={pairwise['n_pairs']})")

        all_results[model_name] = model_results
        with open(os.path.join(args.out_dir, f"{model_name}_windowed_kld.json"), "w") as f:
            json.dump(model_results, f, indent=2)

    with open(os.path.join(args.out_dir, "windowed_kld_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Paper-ready table
    print("\n" + "=" * 78)
    print("WINDOWED PaSST KLD (pairwise mean, symmetric)")
    print("=" * 78)
    header = f"{'Model':<25}"
    for seg in args.segments:
        header += f"{seg*10}-{(seg+1)*10}s".rjust(14)
    header += "  Delta(last-first)"
    print(header)
    print("-" * len(header))
    for model_name in args.models:
        if model_name not in all_results:
            continue
        row = f"{model_name:<25}"
        vals = []
        for seg in args.segments:
            seg_label = f"{seg*10}-{(seg+1)*10}s"
            if seg_label in all_results[model_name]:
                v = all_results[model_name][seg_label]["sym_kld_mean"]
                row += f"{v:>14.4f}"
                vals.append(v)
            else:
                row += f"{'--':>14}"
        if len(vals) >= 2:
            row += f"   {vals[-1] - vals[0]:+.4f}"
        print(row)


if __name__ == "__main__":
    main()