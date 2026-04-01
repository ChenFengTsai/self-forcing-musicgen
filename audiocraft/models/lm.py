import json
import os
from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    _drop_description_condition
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn


logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


# ------------------------------------------------------------------ #
#  Logging helpers                                                     #
# ------------------------------------------------------------------ #

def _resolve_output_dir(output_dir: tp.Optional[str]) -> str:
    """
    Resolve the directory where temporal CE logs are saved.

    Priority order:
      1. Explicit output_dir argument (if not None and not empty)
      2. $AUDIOCRAFT_DORA_DIR/temporal_logs  (if env var is set)
      3. ./temporal_logs  (fallback)
    """
    if output_dir:
        return output_dir
    dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
    if dora_dir:
        return os.path.join(dora_dir, "temporal_logs")
    return "temporal_logs"


def _save_temporal_results(
    output_dir: tp.Optional[str],
    epoch: int,
    mode: str,                   # "rollout" or "normal"
    prefix_len: int,
    per_step_ce: torch.Tensor,   # (T_roll,)
    window_stats: tp.Dict,
    frame_rate: float,
):
    """Append one JSON-lines record (one line per epoch) to
    {output_dir}/{mode}_ce_temporal.jsonl."""
    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    T_roll = per_step_ce.shape[0]
    ce_list = per_step_ce.float().cpu().tolist()
    times_sec = [(prefix_len + t) / frame_rate for t in range(T_roll)]

    record = {
        "epoch": epoch,
        "mode": mode,
        "prefix_len": prefix_len,
        "frame_rate": frame_rate,
        "per_timestep": [
            {
                "frame":    prefix_len + t,
                "time_sec": round(times_sec[t], 4),
                "ce":       round(ce_list[t], 6),
            }
            for t in range(T_roll)
        ],
        "windows": window_stats,
    }

    jpath = os.path.join(output_dir, f"{mode}_ce_temporal.jsonl")
    with open(jpath, "a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ #
#  Shared per-step CE computation                                      #
# ------------------------------------------------------------------ #

# def _compute_per_step_ce(
#     logits: torch.Tensor,       # (B, T, K, vocab)
#     valid_mask: torch.Tensor,   # (B, T, K)
#     targets: torch.Tensor,      # (B, T, K)
# ) -> torch.Tensor:
#     """
#     Returns per-timestep CE averaged over batch (B) and codebooks (K).
#     Shape: (T,)

#     Strategy: compute CE for each (t, k) slice over valid batch items,
#     then average across K. This keeps the time axis intact.
#     """
#     B, T, K, vocab = logits.shape
#     per_step_ce = torch.zeros(T, device=logits.device, dtype=torch.float32)

#     for q in range(K):
#         for t in range(T):
#             mask_bt = valid_mask[:, t, q]          # (B,)
#             if mask_bt.sum() == 0:
#                 continue
#             logits_bt = logits[:, t, q, :][mask_bt]    # (valid_B, vocab)
#             targets_bt = targets[:, t, q][mask_bt]     # (valid_B,)
#             per_step_ce[t] += F.cross_entropy(
#                 logits_bt.float(), targets_bt, reduction="mean"
#             )

#     per_step_ce = per_step_ce / K
#     return per_step_ce

def _compute_per_step_ce(
    logits: torch.Tensor,       # (B, T, K, vocab)
    valid_mask: torch.Tensor,   # (B, T, K)
    targets: torch.Tensor,      # (B, T, K)
) -> torch.Tensor:
    B, T, K, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)

    safe_targets_flat = targets_flat.clamp(min=0, max=vocab - 1)

    loss_all = F.cross_entropy(logits_flat.float(), safe_targets_flat, reduction='none')
    loss_all = loss_all.view(B, T, K)

    mask_float = valid_mask.float()
    masked_loss = loss_all * mask_float

    sum_loss_per_tk = masked_loss.sum(dim=0)
    valid_count_per_tk = mask_float.sum(dim=0).clamp(min=1.0)
    mean_ce_per_tk = sum_loss_per_tk / valid_count_per_tk

    return mean_ce_per_tk.mean(dim=-1)


def _build_window_stats(
    per_step_ce: torch.Tensor,  # (T_roll,)
    prefix_len: int,
    frame_rate: float,
    window_sizes_sec: tp.List[int],
    T_roll: int,
) -> tp.Dict[str, dict]:
    """Slice per_step_ce into temporal windows and return stats."""
    window_stats: tp.Dict[str, dict] = {}
    for w_sec in window_sizes_sec:
        w_frames = int(w_sec * frame_rate)
        end_frame = min(w_frames, T_roll)
        if end_frame == 0:
            continue
        window_ce = per_step_ce[:end_frame].mean().item()
        window_stats[f"{w_sec}s"] = {
            "mean_ce":    round(window_ce, 6),
            "start_frame": prefix_len,
            "end_frame":   prefix_len + end_frame,
            "n_frames":    end_frame,
        }
    return window_stats


# ------------------------------------------------------------------ #
#  Original boilerplate (unchanged)                                    #
# ------------------------------------------------------------------ #

def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    std = 1 / math.sqrt(input_dim)
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)
    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor    # [B, K, T]


class LMModel(StreamingModule):
    def __init__(self, pattern_provider: CodebooksPatternProvider,
                 condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024,
                 dim: int = 128, num_heads: int = 8, hidden_scale: int = 4,
                 norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None,
                 depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0,
                 cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {},
                 two_step_cfg: bool = False, **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

    # ---- all original methods below are UNCHANGED ---- #

    def _init_weights(self, weight_init, depthwise_init, zero_bias_init):
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None
        assert not zero_bias_init or weight_init is not None
        if weight_init is None:
            return
        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)
        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, sequence, conditions, condition_tensors=None, stage=-1):
        B, K, S = sequence.shape
        assert K == self.num_codebooks
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        if condition_tensors is None:
            assert not self._is_streaming
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions
        input_, cross_attention_input = self.fuser(input_, condition_tensors)
        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]
        return logits

    def compute_predictions(self, codes, conditions, condition_tensors=None,
                            stage=-1, keep_only_valid_steps=True):
        B, K, T = codes.shape
        codes = codes.contiguous()
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=keep_only_valid_steps)
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors, stage=stage)
        logits = logits.permute(0, 3, 1, 2)
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps)
        logits = logits.permute(0, 2, 3, 1)
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)
        return LMOutput(logits, logits_mask)

    def _sample_next_token(self, sequence, cfg_conditions, unconditional_state,
                           use_sampling=False, temp=1.0, top_k=0, top_p=0.0,
                           cfg_coef=None, cfg_coef_beta=None, two_step_cfg=None):
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if cfg_coef_beta is not None:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                sequence = torch.cat([sequence, sequence, sequence], dim=0)
            all_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, wav_logits, uncond_logits = all_logits.split(B, dim=0)
                logits = uncond_logits + cfg_coef * (
                    wav_logits + cfg_coef_beta * (cond_logits - wav_logits) - uncond_logits)
        elif two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)
        logits = logits[..., -1]
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    @torch.no_grad()
    def generate(self, prompt=None, conditions=[], num_samples=None,
                 max_gen_len=256, use_sampling=True, temp=1.0, top_k=250,
                 top_p=0.0, cfg_coef=None, cfg_coef_beta=None,
                 two_step_cfg=None, remove_prompts=False, check=False,
                 callback=None):
        assert not self.training
        first_param = next(iter(self.parameters()))
        device = first_param.device
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples]
        num_samples = possible_num_samples[0]

        cfg_conditions: CFGConditions = {}
        if cfg_coef_beta is not None:
            if conditions:
                wav_conditions = _drop_description_condition(conditions)
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + wav_conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        elif conditions:
            two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
            if conditions:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                if two_step_cfg:
                    cfg_conditions = (
                        self.condition_provider(self.condition_provider.tokenize(conditions)),
                        self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                    )
                else:
                    conditions = conditions + null_conditions
                    tokenized = self.condition_provider.tokenize(conditions)
                    cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_codes[..., :start_offset] = prompt
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]
            for offset in range(start_offset_sequence, gen_sequence_len):
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                    assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                    assert not (curr_sequence == unknown_token).any()
                next_token = self._sample_next_token(
                    curr_sequence, cfg_conditions, unconditional_state,
                    use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, cfg_coef_beta=cfg_coef_beta, two_step_cfg=two_step_cfg)
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1])
                prev_offset = offset
                if callback is not None:
                    callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()
        assert not (gen_sequence == unknown_token).any()
        assert (gen_sequence == torch.where(
            mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)).all()
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            gen_sequence, special_token=unknown_token)
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()
        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes

    # @torch.no_grad()
    # def rollout(self, prefix_tokens, condition_tensors, total_len):
    #     """
    #     prefix_tokens: (B, T0, K)
    #     condition_tensors: dict of size-B tensors from condition_provider
    #     total_len: int
    #     """
    #     self.eval()
    #     prompt = prefix_tokens.permute(0, 2, 1)   # (B, K, T0)
    #     B, K, T0 = prompt.shape
    #     device = prompt.device

    #     cfg_conditions = {}
    #     for key, (cond, cond_mask) in condition_tensors.items():
    #         cfg_conditions[key] = (
    #             torch.cat([cond, cond], dim=0),
    #             torch.cat([cond_mask, cond_mask], dim=0),
    #         )

    #     pattern = self.pattern_provider.get_pattern(total_len)
    #     gen_codes = torch.full((B, K, total_len), self.special_token_id, dtype=torch.long, device=device)
    #     gen_codes[..., :T0] = prompt
    #     gen_sequence, _, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
    #     start_offset = pattern.get_first_step_with_timesteps(T0)
    #     assert start_offset is not None

    #     with self.streaming():
    #         unconditional_state = self.get_streaming_state()
    #         prev_offset = 0
    #         for offset in range(start_offset, gen_sequence.shape[-1]):
    #             curr_sequence = gen_sequence[..., prev_offset:offset]
    #             next_token = self._sample_next_token(
    #                 sequence=curr_sequence,
    #                 cfg_conditions=cfg_conditions,
    #                 unconditional_state=unconditional_state,
    #                 use_sampling=False,
    #                 cfg_coef=1.0,
    #             )
    #             valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
    #             next_token[~valid_mask] = self.special_token_id
    #             gen_sequence[..., offset:offset+1] = torch.where(
    #                 gen_sequence[..., offset:offset+1] == self.special_token_id,
    #                 next_token,
    #                 gen_sequence[..., offset:offset+1],
    #             )
    #             prev_offset = offset

    #     unconditional_state.clear()
    #     out_codes, _, _ = pattern.revert_pattern_sequence(gen_sequence, special_token=-1)
    #     return out_codes[..., :total_len].permute(0, 2, 1)   # (B, T, K)
    
    @torch.no_grad()
    def rollout(self, prefix_tokens, condition_tensors, total_len):
        was_training = self.training  # ← save state
        self.eval()
        prompt = prefix_tokens.permute(0, 2, 1)   # (B, K, T0)
        B, K, T0 = prompt.shape
        device = prompt.device

        cfg_conditions = {}
        for key, (cond, cond_mask) in condition_tensors.items():
            cfg_conditions[key] = (
                torch.cat([cond, cond], dim=0),
                torch.cat([cond_mask, cond_mask], dim=0),
            )

        pattern = self.pattern_provider.get_pattern(total_len)
        gen_codes = torch.full((B, K, total_len), self.special_token_id, dtype=torch.long, device=device)
        gen_codes[..., :T0] = prompt
        gen_sequence, _, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        start_offset = pattern.get_first_step_with_timesteps(T0)
        assert start_offset is not None

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            for offset in range(start_offset, gen_sequence.shape[-1]):
                curr_sequence = gen_sequence[..., prev_offset:offset]
                next_token = self._sample_next_token(
                    sequence=curr_sequence,
                    cfg_conditions=cfg_conditions,
                    unconditional_state=unconditional_state,
                    use_sampling=False,
                    cfg_coef=1.0,
                )
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == self.special_token_id,
                    next_token,
                    gen_sequence[..., offset:offset+1],
                )
                prev_offset = offset

        unconditional_state.clear()
        out_codes, _, _ = pattern.revert_pattern_sequence(gen_sequence, special_token=-1)

        if was_training:          # ← restore state
            self.train()

        return out_codes[..., :total_len].permute(0, 2, 1)   # (B, T, K)
    # ------------------------------------------------------------------ #
    #  MODIFIED: prefix_rollout_ce                                        #
    # ------------------------------------------------------------------ #

    # def prefix_rollout_ce(
    #     self,
    #     tokens: torch.Tensor,
    #     condition_tensors: ConditionTensors,
    #     prefix_ratio: float = 0.3,
    #     # ---- temporal analysis flag ----
    #     temporal_analysis: bool = False,
    #     epoch: int = 0,
    #     frame_rate: float = 50.0,
    #     window_sizes_sec: tp.List[int] = [10, 20, 30],
    #     output_dir: tp.Optional[str] = None,
    # ) -> tp.Union[torch.Tensor,
    #               tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict]]:
    #     """
    #     Compute rollout cross-entropy loss.

    #     Args:
    #         tokens (Tensor): Ground-truth tokens (B, T, K).
    #         condition_tensors: Single-batch (B) condition tensors.
    #         prefix_ratio (float): Fraction of T used as the given prefix.
    #         temporal_analysis (bool): If False (default), returns a scalar
    #             loss identical to the original behaviour. If True, also
    #             returns per-timestep CE and window stats, and writes them
    #             to disk.
    #         epoch (int): Current epoch index, used as a row key in logs.
    #         frame_rate (float): Codec frame rate (MusicGen default = 50 Hz).
    #         window_sizes_sec (list[int]): Window boundaries in seconds.
    #         output_dir (str, optional): Override save directory. If None,
    #             falls back to $AUDIOCRAFT_DORA_DIR/temporal_logs, then
    #             ./temporal_logs.

    #     Returns:
    #         If temporal_analysis=False:
    #             scalar_loss (Tensor)
    #         If temporal_analysis=True:
    #             (scalar_loss, per_step_ce, window_stats)
    #               per_step_ce : (T_roll,) float32 CPU tensor
    #               window_stats: dict[str -> dict]
    #     """
    #     B, T, K = tokens.shape
    #     prefix_len = int(T * prefix_ratio)
    #     assert 1 <= prefix_len < T

    #     prefix = tokens[:, :prefix_len]

    #     with torch.no_grad():
    #         with torch.cuda.amp.autocast(enabled=True):
    #             generated = self.rollout(prefix, condition_tensors, T)
    #         # generated: (B, T, K)

    #         gen_input = generated[:, :-1].permute(0, 2, 1)   # (B, K, T-1)

    #         with torch.cuda.amp.autocast(enabled=True):
    #             model_output = self.compute_predictions(gen_input, [], condition_tensors)

    #         logits     = model_output.logits.permute(0, 2, 1, 3)   # (B, T-1, K, vocab)
    #         valid_mask = model_output.mask.permute(0, 2, 1)         # (B, T-1, K)
    #         targets    = tokens[:, 1:]                               # (B, T-1, K)

    #         # restrict to rollout region only
    #         start      = prefix_len - 1
    #         logits     = logits[:, start:]      # (B, T_roll, K, vocab)
    #         valid_mask = valid_mask[:, start:]  # (B, T_roll, K)
    #         targets    = targets[:, start:]     # (B, T_roll, K)
    #         T_roll     = logits.shape[1]

    #         if not temporal_analysis:
    #             # ---- original scalar behaviour (unchanged) ----
    #             loss = 0.0
    #             for q in range(K):
    #                 mask_q   = valid_mask[:, :, q]
    #                 logits_q = logits[:, :, q, :][mask_q]
    #                 targets_q = targets[:, :, q][mask_q]
    #                 if targets_q.numel() == 0:
    #                     continue
    #                 loss += F.cross_entropy(logits_q, targets_q, reduction="mean")
    #             loss = loss / K
    #             return loss

    #         # ---- temporal analysis path ----
    #         per_step_ce = _compute_per_step_ce(logits, valid_mask, targets)
    #         scalar_loss = per_step_ce.mean()

    #         window_stats = _build_window_stats(
    #             per_step_ce, prefix_len, frame_rate, window_sizes_sec, T_roll)

    #         _save_temporal_results(
    #             output_dir=output_dir,
    #             epoch=epoch,
    #             mode="rollout",
    #             prefix_len=prefix_len,
    #             per_step_ce=per_step_ce,
    #             window_stats=window_stats,
    #             frame_rate=frame_rate,
    #         )

    #         return scalar_loss, per_step_ce.cpu(), window_stats

    def prefix_rollout_ce(
            self,
            tokens: torch.Tensor,
            condition_tensors: ConditionTensors,
            prefix_ratio: float = 0.3,
            temporal_analysis: bool = False,
            epoch: int = 0,
            frame_rate: float = 50.0,
            window_sizes_sec: tp.List[int] = [10, 20, 30],
            output_dir: tp.Optional[str] = None,
        ) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict]]:

            B, T, K = tokens.shape
            prefix_len = int(T * prefix_ratio)
            assert 1 <= prefix_len < T

            prefix = tokens[:, :prefix_len]  # (B, prefix_len, K)

            was_training = self.training

            with torch.no_grad():
                # --- rollout: always runs in eval ---
                with torch.cuda.amp.autocast(enabled=True):
                    generated = self.rollout(prefix, condition_tensors, T)
                # generated: (B, T, K) — model is now in eval mode

                # --- compute_predictions: must also run in eval for consistency ---
                gen_input = generated[:, :-1].permute(0, 2, 1)   # (B, K, T-1)

                with torch.cuda.amp.autocast(enabled=True):
                    model_output = self.compute_predictions(gen_input, [], condition_tensors)

                logits     = model_output.logits.permute(0, 2, 1, 3)  # (B, T-1, K, vocab)
                valid_mask = model_output.mask.permute(0, 2, 1)        # (B, T-1, K)
                targets    = tokens[:, 1:]                              # (B, T-1, K) ground-truth

                # restrict to rollout region only
                start      = prefix_len - 1
                logits     = logits[:, start:]
                valid_mask = valid_mask[:, start:]
                targets    = targets[:, start:]
                T_roll     = logits.shape[1]

                # --- NaN guard: check logits before computing loss ---
                if not logits.isfinite().all():
                    if was_training:
                        self.train()
                    
                    # FIX: Return a tuple if temporal_analysis is True to prevent unpacking errors
                    nan_tensor = torch.tensor(float('nan'), device=tokens.device)
                    if temporal_analysis:
                        return nan_tensor, nan_tensor, {}
                    return nan_tensor

                if not temporal_analysis:
                    loss = 0.0
                    for q in range(K):
                        mask_q    = valid_mask[:, :, q].reshape(-1)                                   # bool
                        logits_q  = logits[:, :, q, :].reshape(-1, logits.size(-1))[mask_q].float()  # boolean index + float32
                        targets_q = targets[:, :, q].reshape(-1)[mask_q]
                        if targets_q.numel() == 0:
                            continue
                        loss += F.cross_entropy(logits_q, targets_q)
                    loss = loss / K
                    if was_training:
                        self.train()
                    return loss

                # ---- temporal analysis path ----
                per_step_ce = _compute_per_step_ce(logits, valid_mask, targets)
                scalar_loss = per_step_ce.mean()

                window_stats = _build_window_stats(
                    per_step_ce, prefix_len, frame_rate, window_sizes_sec, T_roll)

                _save_temporal_results(
                    output_dir=output_dir,
                    epoch=epoch,
                    mode="rollout",
                    prefix_len=prefix_len,
                    per_step_ce=per_step_ce,
                    window_stats=window_stats,
                    frame_rate=frame_rate,
                )

            if was_training:
                self.train()

            return scalar_loss, per_step_ce.cpu(), window_stats
    # ------------------------------------------------------------------ #
    #  NEW: compute_normal_ce_temporal                                    #
    # ------------------------------------------------------------------ #

    # def compute_normal_ce_temporal(
    #     self,
    #     tokens: torch.Tensor,
    #     condition_tensors: ConditionTensors,
    #     # ---- temporal analysis flag ----
    #     temporal_analysis: bool = False,
    #     epoch: int = 0,
    #     frame_rate: float = 50.0,
    #     window_sizes_sec: tp.List[int] = [10, 20, 30],
    #     output_dir: tp.Optional[str] = None,
    # ) -> tp.Union[torch.Tensor,
    #               tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict]]:
    #     """
    #     Teacher-forced (normal) cross-entropy, with optional temporal breakdown.

    #     This is the same CE your solver computes each training step, but here
    #     it is optionally decomposed per timestep so you can compare it directly
    #     against the rollout CE to measure exposure-bias magnitude.

    #     Args:
    #         tokens (Tensor): Ground-truth tokens (B, T, K).
    #         condition_tensors: Single-batch (B) condition tensors.
    #         temporal_analysis (bool): If False, returns a scalar CE loss only.
    #             If True, also returns per-timestep CE and window stats, and
    #             writes them to disk.
    #         output_dir (str, optional): Override save directory. If None, falls back to $AUDIOCRAFT_DORA_DIR/temporal_logs.
    #         epoch, frame_rate, window_sizes_sec:
    #             Same meaning as in prefix_rollout_ce.

    #     Returns:
    #         If temporal_analysis=False:
    #             scalar_loss (Tensor)
    #         If temporal_analysis=True:
    #             (scalar_loss, per_step_ce, window_stats)
    #     """
    #     B, T, K = tokens.shape

    #     # Teacher-forced: feed ground-truth tokens shifted by 1
    #     codes_input = tokens[:, :-1].permute(0, 2, 1)   # (B, K, T-1)
    #     targets     = tokens[:, 1:]                       # (B, T-1, K)

    #     with torch.no_grad():
    #         with torch.cuda.amp.autocast(enabled=True):
    #             model_output = self.compute_predictions(codes_input, [], condition_tensors)

    #         logits     = model_output.logits.permute(0, 2, 1, 3)   # (B, T-1, K, vocab)
    #         valid_mask = model_output.mask.permute(0, 2, 1)         # (B, T-1, K)
    #         T_seq      = logits.shape[1]

    #         if not temporal_analysis:
    #             # scalar only
    #             loss = 0.0
    #             for q in range(K):
    #                 mask_q    = valid_mask[:, :, q]
    #                 logits_q  = logits[:, :, q, :][mask_q]
    #                 targets_q = targets[:, :, q][mask_q]
    #                 if targets_q.numel() == 0:
    #                     continue
    #                 loss += F.cross_entropy(logits_q, targets_q, reduction="mean")
    #             loss = loss / K
    #             return loss

    #         # ---- temporal analysis path ----
    #         per_step_ce = _compute_per_step_ce(logits, valid_mask, targets)
    #         scalar_loss = per_step_ce.mean()

    #         # prefix_len=0 for normal CE (no prefix offset)
    #         window_stats = _build_window_stats(
    #             per_step_ce, prefix_len=0, frame_rate=frame_rate,
    #             window_sizes_sec=window_sizes_sec, T_roll=T_seq)

    #         _save_temporal_results(
    #             output_dir=output_dir,
    #             epoch=epoch,
    #             mode="normal",
    #             prefix_len=0,
    #             per_step_ce=per_step_ce,
    #             window_stats=window_stats,
    #             frame_rate=frame_rate,
    #         )

    #         return scalar_loss, per_step_ce.cpu(), window_stats

    def compute_normal_ce_temporal(
            self,
            tokens: torch.Tensor,
            condition_tensors: ConditionTensors,
            temporal_analysis: bool = False,
            epoch: int = 0,
            frame_rate: float = 50.0,
            window_sizes_sec: tp.List[int] = [10, 20, 30],
            output_dir: tp.Optional[str] = None,
        ) -> tp.Union[torch.Tensor,
                    tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict]]:

            B, T, K = tokens.shape

            codes_input = tokens[:, :-1].permute(0, 2, 1)  # (B, K, T-1)
            targets     = tokens[:, 1:]                      # (B, T-1, K)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model_output = self.compute_predictions(codes_input, [], condition_tensors)

                logits     = model_output.logits.permute(0, 2, 1, 3)  # (B, T-1, K, vocab)
                valid_mask = model_output.mask.permute(0, 2, 1)        # (B, T-1, K)
                T_seq      = logits.shape[1]

                if not temporal_analysis:
                    loss = 0.0
                    for q in range(K):
                        mask_q    = valid_mask[:, :, q].reshape(-1)                          # bool
                        logits_q  = logits[:, :, q, :].reshape(-1, logits.size(-1))[mask_q].float()  # boolean index + float32
                        targets_q = targets[:, :, q].reshape(-1)[mask_q]
                        if targets_q.numel() == 0:
                            continue
                        loss += F.cross_entropy(logits_q, targets_q)
                    loss = loss / K
                    return loss

                # ---- temporal analysis path ----
                per_step_ce = _compute_per_step_ce(logits, valid_mask, targets)
                scalar_loss = per_step_ce.mean()

                window_stats = _build_window_stats(
                    per_step_ce, prefix_len=0, frame_rate=frame_rate,
                    window_sizes_sec=window_sizes_sec, T_roll=T_seq)

                _save_temporal_results(
                    output_dir=output_dir,
                    epoch=epoch,
                    mode="normal",
                    prefix_len=0,
                    per_step_ce=per_step_ce,
                    window_stats=window_stats,
                    frame_rate=frame_rate,
                )

                return scalar_loss, per_step_ce.cpu(), window_stats
    # ------------------------------------------------------------------ #
    #  NEW: Structure-Aware Scheduled Sampling (SASS)                     #
    # ------------------------------------------------------------------ #
    def compute_sass_predictions(
            self,
            codes: torch.Tensor,
            conditions: tp.List,
            condition_tensors: tp.Optional[ConditionTensors] = None,
            stage: int = -1,
            keep_only_valid_steps: bool = True,
            # ---- SASS hyperparameters ----
            current_step: int = 0,
            p_max: float = 0.25,
            warmup_start: int = 5000,
            warmup_ramp: int = 8000,
            alpha: float = 2.0,
            beta: float = 0.25,
            gamma: float = 2.0,
            c_min: float = 0.35,
            c_max: float = 0.85,
            token_cap: float = 0.25,
        ) -> LMOutput:
            
            B, K, T = codes.shape
            eps = 1e-8

            # 1. Warmup gate g_warmup(u)
            if current_step < warmup_start:
                g_warmup = 0.0
            else:
                g_warmup = min(1.0, (current_step - warmup_start) / (warmup_ramp + eps))

            # Early exit: if warmup hasn't started, skip SASS entirely to save compute
            if g_warmup == 0.0:
                return self.compute_predictions(
                    codes, conditions, condition_tensors,
                    stage=stage, keep_only_valid_steps=keep_only_valid_steps,
                )

            # ---- Pass 1: teacher-forced (detached and in EVAL mode) ----
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    first_pass = self.compute_predictions(
                        codes, conditions, condition_tensors,
                        stage=stage, keep_only_valid_steps=keep_only_valid_steps,
                    )
                    first_logits = first_pass.logits
                    pred_tokens = first_logits.argmax(dim=-1)           # [B, K, T]
                    
                    # Memory optimization: Cast to float32 only for softmax, then immediately reduce
                    confidence = first_logits.to(torch.float32).softmax(dim=-1).max(dim=-1).values  # [B, K, T]
                    
                    # Explicitly delete references to free memory before Pass 2
                    del first_pass, first_logits
            finally:
                if was_training:
                    self.train()

            # ---- Compute SASS replacement probability p_bkt [B, K, T] ----

            # 2. Temporal gate g_time(t) — [T]
            t_idx = torch.arange(T, device=codes.device, dtype=torch.float32)
            tau = t_idx / (T - 1 + eps)
            g_time = tau.pow(alpha)  # [T]

            # 3. Depth gate g_depth(k) — [K]
            k_idx = torch.arange(K, device=codes.device, dtype=torch.float32)
            delta = k_idx / (K - 1 + eps)
            g_depth = beta + (1.0 - beta) * delta.pow(gamma)  # [K]

            # 4. Confidence gate g_conf(b,k,t) — [B, K, T]
            g_conf = ((confidence - c_min) / (c_max - c_min + eps)).clamp(0.0, 1.0)

            # Combine
            p_bkt = (
                p_max
                * g_warmup
                * g_time.unsqueeze(0).unsqueeze(0)         # [1, 1, T]
                * g_depth.unsqueeze(0).unsqueeze(-1)       # [1, K, 1]
                * g_conf                                   # [B, K, T]
            )

            # Per-position safety cap
            p_bkt = p_bkt.nan_to_num(nan=0.0, posinf=token_cap, neginf=0.0).clamp(min=0.0, max=token_cap)

            # ---- Sample Bernoulli mask ----
            mask_replace = torch.bernoulli(p_bkt).bool()  # [B, K, T]

            # CRITICAL FIX: Do not replace special padding tokens!
            valid_mask = (codes != self.special_token_id)
            mask_replace = mask_replace & valid_mask
            
            # ==========================================
            # --- QUICK SASS LOGGING SNIPPET ---
            # ==========================================
            if current_step % 100 == 0:
                # Prevent division by zero just in case
                valid_count = valid_mask.sum().float().clamp(min=1.0)
                actual_replace_rate = (mask_replace.sum().float() / valid_count).item()
                avg_p = p_bkt[valid_mask].mean().item()
                
                logger.info(
                    f"[SASS] Step {current_step} | "
                    f"Warmup Gate: {g_warmup:.2f} | "
                    f"Target Avg Prob: {avg_p:.3f} | "
                    f"Actual Replaced: {actual_replace_rate:.3f}"
                )
            # ==========================================

            # ---- Build mixed input ----
            mixed_codes = torch.where(mask_replace, pred_tokens, codes)

            # ---- Pass 2: forward on mixed input (with grad) ----
            return self.compute_predictions(
                mixed_codes, conditions, condition_tensors,
                stage=stage, keep_only_valid_steps=keep_only_valid_steps,
            )