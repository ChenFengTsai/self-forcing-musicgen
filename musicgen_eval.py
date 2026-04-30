import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


# ============================================================
# 1. Pipeline — load fine-tuned MusicGen from Dora checkpoint
#    (imports deferred to generate stage only)
# ============================================================
def load_finetuned_musicgen(checkpoint_path: str, device: str = "cuda"):
    import torch
    from audiocraft.models import MusicGen

    # Load base architecture (gets EnCodec + LM skeleton with pretrained weights)
    print("Loading base facebook/musicgen-small ...")
    model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)

    # Load Dora fine-tune checkpoint
    print(f"Loading fine-tune checkpoint: {checkpoint_path}")
    pkg = torch.load(checkpoint_path, map_location=device)

    # Prefer best_state (EMA-smoothed best weights); fall back to model (latest)
    if "best_state" in pkg and pkg["best_state"].get("model"):
        state = pkg["best_state"]["model"]
        print("  using best_state/model")
    else:
        state = pkg["model"]
        print("  using model (latest, not best)")

    missing, unexpected = model.lm.load_state_dict(state, strict=False)
    print(f"Loaded LM. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")

    model.lm.eval()
    return model

def generate_audio(model, prompt: str, output_path: str,
                   duration: float = 30.0, seed: int = 42):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.set_generation_params(
        duration=duration,
        use_sampling=True,
        top_k=250,
        temperature=1.0,
        cfg_coef=3.0,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    wav = model.generate([prompt], progress=False)  # (1, C, T)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - t0

    audio = wav[0].cpu().numpy()
    sf.write(output_path, audio.T, model.sample_rate)

    rtf = inference_time / duration
    return inference_time, rtf


# ============================================================
# 2. Generate (with RTF logging)
# ============================================================
def generate_for_dataset(model, jsonl_path, generated_dir, rtf_log_path,
                         duration=30.0, max_samples=None):
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(os.path.dirname(rtf_log_path) or ".", exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if max_samples:
        samples = samples[:max_samples]
    print(f"Loaded {len(samples)} samples")

    new_file = not os.path.exists(rtf_log_path)
    csv_f = open(rtf_log_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["filename", "duration_sec", "inference_sec", "rtf"])

    rtfs = []
    for i, s in enumerate(samples):
        prompt = s["main_caption"]
        stem = Path(s["location"]).stem
        out_path = os.path.join(generated_dir, f"{stem}.wav")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(samples)}] skip {stem}")
            continue

        print(f"[{i+1}/{len(samples)}] {stem}: {prompt[:80]}...")
        try:
            inf_time, rtf = generate_audio(model, prompt, out_path, duration=duration)
            rtfs.append(rtf)
            writer.writerow([f"{stem}.wav", duration, f"{inf_time:.4f}", f"{rtf:.4f}"])
            csv_f.flush()
            print(f"   inference={inf_time:.2f}s  RTF={rtf:.3f}")
        except Exception as e:
            print(f"  ! failed: {e}")

    csv_f.close()
    if rtfs:
        print(f"\nMean RTF: {np.mean(rtfs):.3f}  |  Median: {np.median(rtfs):.3f}  |  N={len(rtfs)}")
    return rtfs


# ============================================================
# 3. Chop 30s -> three 10s segments
# ============================================================
def chop_directory(src_dir: str, dst_dir: str, segment_sec: float = 10.0):
    os.makedirs(dst_dir, exist_ok=True)
    for wav_path in Path(src_dir).glob("*.wav"):
        audio, sr = sf.read(wav_path, always_2d=True)
        seg_len = int(segment_sec * sr)
        n_segs = len(audio) // seg_len
        for k in range(n_segs):
            seg = audio[k * seg_len : (k + 1) * seg_len]
            out = Path(dst_dir) / f"{wav_path.stem}_seg{k}.wav"
            if not out.exists():
                sf.write(out, seg, sr)
    print(f"Chopped {src_dir} -> {dst_dir}")


# ============================================================
# 4. FAD (imports deferred to score stage only)
# ============================================================
def compute_fad(reference_dir, generated_dir, model_name="vggish"):
    from fadtk.fad import FrechetAudioDistance
    from fadtk.model_loader import get_all_models

    models = {m.name: m for m in get_all_models()}
    model = models[model_name]

    fad = FrechetAudioDistance(model)
    for d in (reference_dir, generated_dir):
        for wav in Path(d).glob("*.wav"):
            fad.cache_embedding_file(wav)

    score = fad.score(reference_dir, generated_dir)
    print(f"FAD ({model_name}): {score:.4f}")
    return score


# ============================================================
# 5. Run
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample",
                        help="'sample' = N songs (quick test), 'full' = entire dataset")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all",
                        help="'generate' = gen+chop (audiocraft env), "
                             "'score' = FAD only (fadtk env), "
                             "'all' = both (one env)")
    parser.add_argument("--checkpoint",
        default="/storage/ssd3/richtsai1103/Jamendo/teacher/20260403_010611_teacher/xps/37b09569/checkpoint.th")
    parser.add_argument("--fad-model", default="vggish")
    args = parser.parse_args()

    JSONL_PATH    = "/storage/ssd1/richtsai1103/MusicBench/MusicBench_test_A.json"
    REFERENCE_DIR = "/storage/ssd3/richtsai1103/MusicBench/SOA/reference"
    ROOT          = "/storage/ssd3/richtsai1103/MusicBench/MusicGen_TF"

    if args.mode == "sample":
        max_samples       = args.sample_size
        GENERATED_DIR     = f"{ROOT}/generation_sample"
        GENERATED_CHOPPED = f"{ROOT}/generation_sample_10s"
        RTF_LOG_PATH      = f"{ROOT}/rtf_log_sample.csv"
        print(f"=== SAMPLE MODE: {max_samples} songs | stage={args.stage} ===")
    else:
        max_samples       = None
        GENERATED_DIR     = f"{ROOT}/generation"
        GENERATED_CHOPPED = f"{ROOT}/generation_10s"
        RTF_LOG_PATH      = f"{ROOT}/rtf_log.csv"
        print(f"=== FULL MODE: entire dataset | stage={args.stage} ===")

    if args.stage in ("generate", "all"):
        model = load_finetuned_musicgen(args.checkpoint)
        generate_for_dataset(model, JSONL_PATH, GENERATED_DIR, RTF_LOG_PATH,
                             duration=30.0, max_samples=max_samples)
        chop_directory(GENERATED_DIR, GENERATED_CHOPPED, segment_sec=10.0)

    if args.stage in ("score", "all"):
        compute_fad(REFERENCE_DIR, GENERATED_CHOPPED, model_name=args.fad_model)