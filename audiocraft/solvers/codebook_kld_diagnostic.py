# codebook_kld_diagnostic.py  —  FIXED VERSION
# -----------------------------------------------------------------------------
# Fix: windows are now indexed against ABSOLUTE audio time, not relative-to-
# rollout-start time. This matches how your paper defines "0-10s", "10-20s",
# "20-30s" elsewhere (e.g. in windowed FAD, Figure 1c, Tables 1 & 2).
#
# The prior version stripped the prefix from `generated` and `ground_truth`
# before handing them to the accumulator, so the accumulator's "0-10s" was
# actually ~prefix_len/frame_rate to (prefix_len/frame_rate + 10) seconds of
# absolute time. With a 9s prefix and 30s clips, this resulted in only 1s of
# real rollout data landing in the "20-30s" bucket (= 27,500 tokens instead
# of 275,000).
#
# FIXES in this version:
#   1. Accumulator accepts full (B, T, K) tensors AND a prefix_len argument.
#   2. Frames before prefix_len are excluded from histograms (they're ground
#      truth, not self-generated — counting them would wash out the signal).
#   3. Window labels now unambiguously refer to absolute audio time.
#
# Call pattern from eval loop:
#     accumulator.update(
#         generated_tokens=generated,       # (B, 1500, K) — FULL tensor
#         ground_truth_tokens=ground_truth, # (B, 1500, K) — FULL tensor
#         prefix_len=450,                   # frames to exclude from histograms
#     )
# -----------------------------------------------------------------------------

import json
import math
import os
import typing as tp

import torch


class CodebookKLDAccumulator:
    """
    Per-codebook, per-absolute-time-window token histograms.

    Windows are defined in seconds of absolute audio time (from the start of
    the clip). Frames before prefix_len are excluded from accumulation so the
    histograms only reflect self-generated (post-prefix) content.
    """

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        frame_rate: float = 50.0,
        window_bounds_sec: tp.Sequence[tp.Tuple[int, int]] = ((0, 10), (10, 20), (20, 30)),
        special_token_ids: tp.Optional[tp.Sequence[int]] = None,
    ):
        self.K = num_codebooks
        self.V = vocab_size
        self.frame_rate = frame_rate
        self.windows = list(window_bounds_sec)
        self.special_token_ids = set(special_token_ids or [])

        self.hist_gen: tp.Dict[tp.Tuple[int, int], torch.Tensor] = {}
        self.hist_gt: tp.Dict[tp.Tuple[int, int], torch.Tensor] = {}

        for k in range(self.K):
            for w_idx in range(len(self.windows)):
                self.hist_gen[(k, w_idx)] = torch.zeros(self.V, dtype=torch.long)
                self.hist_gt[(k, w_idx)] = torch.zeros(self.V, dtype=torch.long)

    def _update_hist(
        self,
        tokens: torch.Tensor,                   # (B, T, K) — FULL clip length
        hist_dict: tp.Dict[tp.Tuple[int, int], torch.Tensor],
        prefix_len: int,                        # frames to exclude (absolute)
    ):
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B, T, K), got {tokens.shape}")
        B, T, K = tokens.shape
        if K != self.K:
            raise ValueError(f"K mismatch: accumulator expects {self.K}, got {K}")

        tokens_cpu = tokens.detach().to(torch.long).cpu()

        for w_idx, (s_sec, e_sec) in enumerate(self.windows):
            s_frame = int(s_sec * self.frame_rate)
            e_frame = int(e_sec * self.frame_rate)
            e_frame = min(e_frame, T)

            # Exclude prefix frames: bump s_frame up past the prefix if needed.
            s_frame = max(s_frame, prefix_len)

            if s_frame >= e_frame:
                # This window lies entirely within the prefix → no rollout
                # data to accumulate for this window.
                continue

            window_slice = tokens_cpu[:, s_frame:e_frame, :]

            for k in range(self.K):
                flat = window_slice[:, :, k].reshape(-1)

                valid = (flat >= 0) & (flat < self.V)
                if self.special_token_ids:
                    for sid in self.special_token_ids:
                        valid &= flat != sid

                flat_valid = flat[valid]
                if flat_valid.numel() == 0:
                    continue

                counts = torch.bincount(flat_valid, minlength=self.V)
                hist_dict[(k, w_idx)] += counts

    def update(
        self,
        generated_tokens: torch.Tensor,     # (B, T, K) FULL clip
        ground_truth_tokens: torch.Tensor,  # (B, T, K) FULL clip
        prefix_len: int,                    # FRAMES to exclude from histograms
    ):
        """Add one eval batch's tokens to the running histograms.

        Unlike the prior version, pass the FULL (B, T, K) tensors here —
        do NOT strip the prefix beforehand. The accumulator will exclude
        frames [0, prefix_len) internally so the window labels retain their
        absolute-audio-time meaning.
        """
        self._update_hist(generated_tokens, self.hist_gen, prefix_len)
        self._update_hist(ground_truth_tokens, self.hist_gt, prefix_len)

    @staticmethod
    def _kld(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
        p = p.double()
        q = q.double()
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum <= 0 or q_sum <= 0:
            return float("nan")
        p = p / p_sum
        q = q / q_sum
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        return float((p * (p.log() - q.log())).sum().item())

    def compute(self) -> tp.Dict[str, tp.Any]:
        out = {"per_codebook_window": {}, "meta": {
            "num_codebooks": self.K,
            "vocab_size": self.V,
            "frame_rate": self.frame_rate,
            "windows_sec": [list(w) for w in self.windows],
            "window_convention": "absolute_audio_time",
        }}

        for k in range(self.K):
            key_k = f"q{k+1}"
            out["per_codebook_window"][key_k] = {}
            for w_idx, (s, e) in enumerate(self.windows):
                win_key = f"{s}-{e}s"
                p_gen = self.hist_gen[(k, w_idx)]
                p_gt = self.hist_gt[(k, w_idx)]

                fwd = self._kld(p_gen, p_gt)
                rev = self._kld(p_gt, p_gen)
                sym = 0.5 * (fwd + rev) if not (math.isnan(fwd) or math.isnan(rev)) else float("nan")

                out["per_codebook_window"][key_k][win_key] = {
                    "fwd_kld": round(fwd, 6) if not math.isnan(fwd) else None,
                    "rev_kld": round(rev, 6) if not math.isnan(rev) else None,
                    "sym_kld": round(sym, 6) if not math.isnan(sym) else None,
                    "n_gen_tokens": int(p_gen.sum().item()),
                    "n_gt_tokens":  int(p_gt.sum().item()),
                }

        return out

    def finalize_and_save(
        self,
        output_dir: str,
        tag: str,
        epoch: int = 0,
    ) -> tp.Dict[str, tp.Any]:
        os.makedirs(output_dir, exist_ok=True)
        result = self.compute()
        record = {"epoch": epoch, "tag": tag, **result}
        path = os.path.join(output_dir, f"codebook_kld_{tag}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return result


@torch.no_grad()
def rollout_and_accumulate(
    lm_model,
    ground_truth_tokens: torch.Tensor,  # (B, T, K) — FULL clip
    condition_tensors,
    accumulator: CodebookKLDAccumulator,
    prefix_ratio: float = 0.3,
):
    """Self-generate from a prefix, then update the accumulator with the FULL
    (B, T, K) tensors (not the post-prefix slice). The accumulator excludes
    prefix frames internally so window labels are in absolute audio time."""
    B, T, K = ground_truth_tokens.shape
    prefix_len = max(1, int(T * prefix_ratio))
    prefix = ground_truth_tokens[:, :prefix_len]

    was_training = lm_model.training
    lm_model.eval()
    try:
        with torch.cuda.amp.autocast(enabled=True):
            generated = lm_model.rollout(prefix, condition_tensors, T)  # (B, T, K)
    finally:
        if was_training:
            lm_model.train()

    # Pass the FULL tensors, not gen_roll / gt_roll.
    accumulator.update(
        generated_tokens=generated,
        ground_truth_tokens=ground_truth_tokens,
        prefix_len=prefix_len,
    )
    return generated