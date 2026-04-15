"""
Standalone test for the FAD stdout parsing bug in audiocraft/metrics/fad.py.

Usage:
    # 1. Activate the same conda env you use for training:
    conda activate audio

    # 2. Make sure PYTHONPATH points at google-research (same as your working manual run):
    export PYTHONPATH=/home/richtsai1103/CRL/audiocraft/google-research:$PYTHONPATH

    # 3. Run:
    python test_fad_parse.py \
        --test_stats    /storage/ssd1/richtsai1103/Jamendo/dora_outputs/20260414_000052/xps/c1fc3463/fad/stats_tests \
        --background_stats /storage/ssd1/richtsai1103/Jamendo/dora_outputs/20260414_000052/xps/c1fc3463/fad/stats_background

It will:
  1. Launch `python -m frechet_audio_distance.compute_fad ...` exactly like audiocraft does.
  2. Show you the raw stdout bytes.
  3. Try the BROKEN parse (`float(result.stdout[4:])`) — should raise.
  4. Try the FIXED parse (regex on decoded stdout) — should print 2.661762.
"""
import argparse
import os
import re
import subprocess
import sys


def broken_parse(stdout_bytes: bytes) -> float:
    # This is literally what audiocraft/metrics/fad.py does today.
    return float(stdout_bytes[4:])


def fixed_parse(stdout_bytes: bytes) -> float:
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    match = re.search(r"FAD:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout_text)
    if match is None:
        raise RuntimeError(f"Could not find FAD score in stdout:\n{stdout_text}")
    return float(match.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_stats", required=True)
    ap.add_argument("--background_stats", required=True)
    ap.add_argument(
        "--python_exe",
        default=os.environ.get("TF_PYTHON_EXE", "python"),
        help="Same python audiocraft uses (TF_PYTHON_EXE or 'python').",
    )
    ap.add_argument(
        "--quiet_tf",
        action="store_true",
        help="Set TF_CPP_MIN_LOG_LEVEL=3 to silence TF stdout chatter.",
    )
    args = ap.parse_args()

    cmd = [
        args.python_exe,
        "-m",
        "frechet_audio_distance.compute_fad",
        "--test_stats",
        args.test_stats,
        "--background_stats",
        args.background_stats,
    ]
    print(f"[*] Running: {' '.join(cmd)}\n")

    env = os.environ.copy()
    if args.quiet_tf:
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    result = subprocess.run(cmd, env=env, capture_output=True)

    print(f"[*] returncode: {result.returncode}")
    print(f"[*] stdout length: {len(result.stdout)} bytes")
    print(f"[*] stderr length: {len(result.stderr)} bytes")
    print("\n===== RAW STDOUT =====")
    sys.stdout.buffer.write(result.stdout)
    print("\n===== END STDOUT =====\n")

    if result.returncode != 0:
        print("[!] Non-zero return code. stderr:")
        sys.stdout.buffer.write(result.stderr)
        sys.exit(1)

    print("----- First 4 bytes of stdout (what `[4:]` skips) -----")
    print(repr(result.stdout[:4]))
    print()

    print("----- BROKEN parse: float(result.stdout[4:]) -----")
    try:
        score = broken_parse(result.stdout)
        print(f"   got: {score}  (no error — TF was quiet enough)")
    except Exception as e:
        print(f"   FAILED as expected: {type(e).__name__}: {e}")
    print()

    print("----- FIXED parse: regex on decoded stdout -----")
    try:
        score = fixed_parse(result.stdout)
        print(f"   FAD score = {score}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()