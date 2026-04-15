# # # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# from pathlib import Path
# import json
# import os
# import time
# import typing as tp
# import warnings

# import flashy
# import librosa
# import math
# import numpy as np
# import omegaconf
# import torch
# from torch.nn import functional as F

# from . import base, builders
# from .compression import CompressionSolver
# from .. import metrics as eval_metrics
# from .. import models
# from ..data.audio_dataset import AudioDataset
# from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
# from ..data.audio_utils import normalize_audio
# from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, \
#             StyleConditioner, _drop_description_condition
# from ..utils.cache import CachedBatchWriter, CachedBatchLoader
# from ..utils.samples.manager import SampleManager
# from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash


# class MusicGenSolver(base.StandardSolver):
#     """Solver for MusicGen training task.

#     Used in: https://arxiv.org/abs/2306.05284
#     """
#     DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

#     def __init__(self, cfg: omegaconf.DictConfig):
#         super().__init__(cfg)
#         # easier access to sampling parameters
#         self.generation_params = {
#             'use_sampling': self.cfg.generate.lm.use_sampling,
#             'temp': self.cfg.generate.lm.temp,
#             'top_k': self.cfg.generate.lm.top_k,
#             'top_p': self.cfg.generate.lm.top_p,
#         }
#         self._best_metric_name: tp.Optional[str] = 'ce'

#         self._cached_batch_writer = None
#         self._cached_batch_loader = None
#         if cfg.cache.path:
#             if cfg.cache.write:
#                 self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
#                 if self.cfg.cache.write_num_shards:
#                     self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
#                     self._best_metric_name = None
#             else:
#                 self._cached_batch_loader = CachedBatchLoader(
#                     Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
#                     min_length=self.cfg.optim.updates_per_epoch or 1)
#                 self.dataloaders['original_train'] = self.dataloaders['train']
#                 self.dataloaders['train'] = self._cached_batch_loader  # type: ignore

#     @staticmethod
#     def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
#                                  device: tp.Optional[str] = None, autocast: bool = True,
#                                  batch_size: tp.Optional[int] = None,
#                                  override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
#                                  **kwargs):
#         """Mostly a convenience function around magma.train.get_solver_from_sig,
#         populating all the proper param, deactivating EMA, FSDP, loading the best state,
#         basically all you need to get a solver ready to "play" with in single GPU mode
#         and with minimal memory overhead.

#         Args:
#             sig (str): signature to load.
#             dtype (str or None): potential dtype, as a string, i.e. 'float16'.
#             device (str or None): potential device, as a string, i.e. 'cuda'.
#             override_cfg (dict or omegaconf.DictConfig or None): potential device, as a string, i.e. 'cuda'.
#         """
#         from audiocraft import train
#         our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
#         our_override_cfg['autocast'] = autocast
#         if dtype is not None:
#             our_override_cfg['dtype'] = dtype
#         if device is not None:
#             our_override_cfg['device'] = device
#         if batch_size is not None:
#             our_override_cfg['dataset'] = {'batch_size': batch_size}
#         if override_cfg is None:
#             override_cfg = {}
#         override_cfg = omegaconf.OmegaConf.merge(
#             omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))  # type: ignore
#         solver = train.get_solver_from_sig(
#             sig, override_cfg=override_cfg,
#             load_best=True, disable_fsdp=True,
#             ignore_state_keys=['optimizer', 'ema'], **kwargs)
#         solver.model.eval()
#         return solver

#     def get_formatter(self, stage_name: str) -> flashy.Formatter:
#         return flashy.Formatter({
#             'lr': '.2E',
#             'ce': '.3f',
#             'ppl': '.3f',
#             'grad_norm': '.3E',
#         }, exclude_keys=['ce_q*', 'ppl_q*'])

#     @property
#     def best_metric_name(self) -> tp.Optional[str]:
#         return self._best_metric_name

#     def initialize_optimization(self) -> None:
#         if self.cfg.fsdp.use:
#             assert not self.cfg.autocast, "Cannot use autocast with fsdp"
#             self.model = self.wrap_with_fsdp(self.model)
#         self.register_ema('model')
#         # initialize optimization
#         self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
#         self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
#         self.register_stateful('model', 'optimizer', 'lr_scheduler')
#         self.register_best_state('model')
#         self.autocast_dtype = {
#             'float16': torch.float16, 'bfloat16': torch.bfloat16
#         }[self.cfg.autocast_dtype]
#         self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
#         if self.cfg.fsdp.use:
#             need_scaler = self.cfg.fsdp.param_dtype == 'float16'
#         else:
#             need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
#         if need_scaler:
#             if self.cfg.fsdp.use:
#                 from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
#                 self.scaler = ShardedGradScaler()  # type: ignore
#             else:
#                 self.scaler = torch.cuda.amp.GradScaler()
#             self.register_stateful('scaler')

#     def build_model(self) -> None:
#         """Instantiate models and optimizer."""
#         # we can potentially not use all quantizers with which the EnCodec model was trained
#         # (e.g. we trained the model with quantizers dropout)
#         self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
#             self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
#         assert self.compression_model.sample_rate == self.cfg.sample_rate, (
#             f"Compression model sample rate is {self.compression_model.sample_rate} but "
#             f"Solver sample rate is {self.cfg.sample_rate}."
#             )
#         # ensure we have matching configuration between LM and compression model
#         assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
#             "Cardinalities of the LM and compression model don't match: ",
#             f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
#             f"compression model cardinality is {self.compression_model.cardinality}"
#         )
#         assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
#             "Numbers of codebooks of the LM and compression models don't match: ",
#             f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
#             f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
#         )
#         self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
#                          self.compression_model.num_codebooks, self.compression_model.cardinality,
#                          self.compression_model.frame_rate)
#         # instantiate LM model
#         self.model: tp.Union[models.LMModel, models.FlowMatchingModel] = models.builders.get_lm_model(
#             self.cfg).to(self.device)

#         # initialize optimization
#         self.initialize_optimization()

#     def build_dataloaders(self) -> None:
#         """Instantiate audio dataloaders for each stage."""
#         self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

#     def show(self) -> None:
#         """Show the compression model and LM model."""
#         self.logger.info("Compression model:")
#         self.log_model_summary(self.compression_model)
#         self.logger.info("LM model:")
#         self.log_model_summary(self.model)

#     def load_state_dict(self, state: dict) -> None:
#         if 'condition_provider' in state:
#             model_state = state['model']
#             condition_provider_state = state.pop('condition_provider')
#             prefix = 'condition_provider.'
#             for key, value in condition_provider_state.items():
#                 key = prefix + key
#                 assert key not in model_state
#                 model_state[key] = value
#         if 'compression_model' in state:
#             # We used to store the `compression_model` state in the checkpoint, however
#             # this is in general not needed, as the compression model should always be readable
#             # from the original `cfg.compression_model_checkpoint` location.
#             compression_model_state = state.pop('compression_model')
#             before_hash = model_hash(self.compression_model)
#             self.compression_model.load_state_dict(compression_model_state)
#             after_hash = model_hash(self.compression_model)
#             if before_hash != after_hash:
#                 raise RuntimeError(
#                     "The compression model state inside the checkpoint is different"
#                     " from the one obtained from compression_model_checkpoint..."
#                     "We do not support altering the compression model inside the LM "
#                     "checkpoint as parts of the code, in particular for running eval post-training "
#                     "will use the compression_model_checkpoint as the source of truth.")

#         super().load_state_dict(state)

#     def load_from_pretrained(self, name: str):
#         # TODO: support native HF versions of MusicGen.
#         lm_pkg = models.loaders.load_lm_model_ckpt(name)
#         state: dict = {
#             'best_state': {
#                 'model': lm_pkg['best_state'],
#             },
#         }
#         return state

#     # def _compute_cross_entropy(
#     #     self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
#     # ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
#     #     """Compute cross entropy between multi-codebook targets and model's logits.
#     #     The cross entropy is computed per codebook to provide codebook-level cross entropy.
#     #     Valid timesteps for each of the codebook are pulled from the mask, where invalid
#     #     timesteps are set to 0.

#     #     Args:
#     #         logits (torch.Tensor): Model's logits of shape [B, K, T, card].
#     #         targets (torch.Tensor): Target codes, of shape [B, K, T].
#     #         mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
#     #     Returns:
#     #         ce (torch.Tensor): Cross entropy averaged over the codebooks
#     #         ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
#     #     """
#     #     B, K, T = targets.shape
#     #     assert logits.shape[:-1] == targets.shape
#     #     assert mask.shape == targets.shape
#     #     ce = torch.zeros([], device=targets.device)
#     #     ce_per_codebook: tp.List[torch.Tensor] = []
#     #     for k in range(K):
#     #         logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
#     #         targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
#     #         mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
#     #         ce_targets = targets_k[mask_k]
#     #         ce_logits = logits_k[mask_k]
#     #         q_ce = F.cross_entropy(ce_logits, ce_targets)
#     #         ce += q_ce
#     #         ce_per_codebook.append(q_ce.detach())
#     #     # average cross entropy across codebooks
#     #     ce = ce / K
#     #     return ce, ce_per_codebook
    
#     def _compute_cross_entropy(
#             self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
#         ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
#             B, K, T = targets.shape
#             assert logits.shape[:-1] == targets.shape
#             assert mask.shape == targets.shape
#             ce = torch.zeros([], device=targets.device, dtype=torch.float32)
#             ce_per_codebook: tp.List[torch.Tensor] = []
#             for k in range(K):
#                 logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
#                 targets_k = targets[:, k, ...].contiguous().view(-1)                 # [B x T]
#                 mask_k = mask[:, k, ...].contiguous().view(-1)                       # [B x T] bool
#                 ce_targets = targets_k[mask_k]
#                 ce_logits = logits_k[mask_k].float()  # boolean index excludes padding, float32 prevents fp16 overflow
#                 if ce_targets.numel() == 0:
#                     continue
#                 q_ce = F.cross_entropy(ce_logits, ce_targets)
#                 ce += q_ce
#                 ce_per_codebook.append(q_ce.detach())
#             ce = ce / K
#             return ce, ce_per_codebook

#     def _get_audio_tokens(self, audio: torch.Tensor):
#         with torch.no_grad():
#             audio_tokens, scale = self.compression_model.encode(audio)
#             assert scale is None, "Scaled compression model not supported with LM."
#             return audio_tokens

#     def _prepare_tokens_and_attributes(
#         self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
#         check_synchronization_points: bool = False
#     ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
#         """Prepare input batchs for language model training.

#         Args:
#             batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
#                 and corresponding metadata as SegmentWithAttributes (with B items).
#             check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
#         Returns:
#             Condition tensors (dict[str, any]): Preprocessed condition attributes.
#             Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
#                 with B the batch size, K the number of codebooks, T_s the token timesteps.
#             Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
#         """
#         if self.model.training:
#             warnings.warn(
#                 "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
#                 "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
#                 "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
#                 "Really sorry about that.")
#         if self._cached_batch_loader is None or self.current_stage != "train":
#             audio, infos = batch
#             audio = audio.to(self.device)
#             audio_tokens = None
#             assert audio.size(0) == len(infos), (
#                 f"Mismatch between number of items in audio batch ({audio.size(0)})",
#                 f" and in metadata ({len(infos)})"
#             )
#         else:
#             audio = None
#             # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
#             infos, = batch  # type: ignore
#             assert all([isinstance(info, AudioInfo) for info in infos])
#             assert all([info.audio_tokens is not None for info in infos])  # type: ignore
#             audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
#             audio_tokens = audio_tokens.long()
#             for info in infos:
#                 if isinstance(info, MusicInfo):
#                     # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
#                     # then you must be using the chroma cache! otherwise the code will try
#                     # to use this segment and fail (by that I mean you will see NaN everywhere).
#                     info.self_wav = WavCondition(
#                         torch.full([1, info.channels, info.total_frames], float('NaN')),
#                         length=torch.tensor([info.n_frames]),
#                         sample_rate=[info.sample_rate],
#                         path=[info.meta.path],
#                         seek_time=[info.seek_time])
#                     dataset = get_dataset_from_loader(self.dataloaders['original_train'])
#                     assert isinstance(dataset, MusicDataset), type(dataset)
#                     if dataset.paraphraser is not None and info.description is not None:
#                         # Hackingly reapplying paraphraser when using cache.
#                         info.description = dataset.paraphraser.sample_paraphrase(
#                             info.meta.path, info.description)
#         # prepare attributes
#         attributes = [info.to_condition_attributes() for info in infos]
#         attributes = self.model.cfg_dropout(attributes)
#         attributes = self.model.att_dropout(attributes)
#         tokenized = self.model.condition_provider.tokenize(attributes)

#         # Now we should be synchronization free.
#         if self.device == "cuda" and check_synchronization_points:
#             torch.cuda.set_sync_debug_mode("warn")

#         if audio_tokens is None:
#             audio_tokens = self._get_audio_tokens(audio)

#         with self.autocast:
#             condition_tensors = self.model.condition_provider(tokenized)

#         # create a padding mask to hold valid vs invalid positions
#         padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
#         # replace encodec tokens from padded audio with special_token_id
#         if self.cfg.tokens.padding_with_special_token:
#             audio_tokens = audio_tokens.clone()
#             padding_mask = padding_mask.clone()
#             token_sample_rate = self.compression_model.frame_rate
#             B, K, T_s = audio_tokens.shape
#             for i in range(B):
#                 n_samples = infos[i].n_frames
#                 audio_sample_rate = infos[i].sample_rate
#                 # take the last token generated from actual audio frames (non-padded audio)
#                 valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
#                 audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
#                 padding_mask[i, :, valid_tokens:] = 0

#         if self.device == "cuda" and check_synchronization_points:
#             torch.cuda.set_sync_debug_mode("default")

#         if self._cached_batch_writer is not None and self.current_stage == 'train':
#             assert self._cached_batch_loader is None
#             assert audio_tokens is not None
#             for info, one_audio_tokens in zip(infos, audio_tokens):
#                 assert isinstance(info, AudioInfo)
#                 if isinstance(info, MusicInfo):
#                     assert not info.joint_embed, "joint_embed and cache not supported yet."
#                     info.self_wav = None
#                 assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
#                 info.audio_tokens = one_audio_tokens.short().cpu()
#             self._cached_batch_writer.save(infos)

#         return condition_tensors, audio_tokens, padding_mask

#     def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
#         """Perform one training or valid step on a given batch."""
#         check_synchronization_points = idx == 1 and self.device == 'cuda'

#         condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
#             batch, check_synchronization_points)

#         self.deadlock_detect.update('tokens_and_conditions')

#         if check_synchronization_points:
#             torch.cuda.set_sync_debug_mode('warn')

#         with self.autocast:
#             style_mask = None
#             if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
#                 if isinstance(self.model.condition_provider.conditioners.self_wav, StyleConditioner):
#                     style_mask = self.model.condition_provider.conditioners.self_wav.mask

#             # ---- SASS: use structure-aware scheduled sampling if enabled ----
#             use_sass = self.is_training and getattr(self.cfg, 'sass', None) is not None \
#                        and getattr(self.cfg.sass, 'enabled', False)
#             # print(use_sass)
#             if use_sass:
#                 sass_cfg = self.cfg.sass
#                 steps_per_epoch = len(self.dataloaders['train'])
#                 current_step = (self.epoch - 1) * steps_per_epoch + idx
#                 # print('Step', current_step)
#                 model_output = self.model.compute_sass_predictions(
#                     audio_tokens, [], condition_tensors,
#                     current_step=current_step,
#                     p_max=getattr(sass_cfg, 'p_max', 0.25),
#                     warmup_start=getattr(sass_cfg, 'warmup_start', 5000),
#                     warmup_ramp=getattr(sass_cfg, 'warmup_ramp', 8000),
#                     alpha=getattr(sass_cfg, 'alpha', 2.0),
#                     beta=getattr(sass_cfg, 'beta', 0.25),
#                     gamma=getattr(sass_cfg, 'gamma', 2.0),
#                     c_min=getattr(sass_cfg, 'c_min', 0.35),
#                     c_max=getattr(sass_cfg, 'c_max', 0.85),
#                     token_cap=getattr(sass_cfg, 'token_cap', 0.25),
#                 )  # type: ignore

#             else:
#                 model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
#             logits = model_output.logits
#             if style_mask is not None:
#                 mask = padding_mask & model_output.mask & style_mask
#             else:
#                 mask = padding_mask & model_output.mask
#             ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
#             loss = ce
#         self.deadlock_detect.update('loss')

#         if check_synchronization_points:
#             torch.cuda.set_sync_debug_mode('default')

#         if self.is_training:
#             metrics['lr'] = self.optimizer.param_groups[0]['lr']
#             if self.scaler is not None:
#                 loss = self.scaler.scale(loss)
#             self.deadlock_detect.update('scale')
#             if self.cfg.fsdp.use:
#                 loss.backward()
#                 flashy.distrib.average_tensors(self.model.buffers())
#             elif self.cfg.optim.eager_sync:
#                 with flashy.distrib.eager_sync_model(self.model):
#                     loss.backward()
#             else:
#                 # this should always be slower but can be useful
#                 # for weird use cases like multiple backwards.
#                 loss.backward()
#                 flashy.distrib.sync_model(self.model)
#             self.deadlock_detect.update('backward')

#             if self.scaler is not None:
#                 self.scaler.unscale_(self.optimizer)
#             if self.cfg.optim.max_norm:
#                 if self.cfg.fsdp.use:
#                     metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
#                 else:
#                     metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
#                         self.model.parameters(), self.cfg.optim.max_norm
#                     )
#             if self.scaler is None:
#                 self.optimizer.step()
#             else:
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#             if self.lr_scheduler:
#                 self.lr_scheduler.step()
#             self.optimizer.zero_grad()
#             self.deadlock_detect.update('optim')
#             if self.scaler is not None:
#                 scale = self.scaler.get_scale()
#                 metrics['grad_scale'] = scale
#             if not loss.isfinite().all():
#                 raise RuntimeError("Model probably diverged.")

#         metrics['ce'] = ce
#         metrics['ppl'] = torch.exp(ce)
        
#         # ===== Prefix rollout CE + temporal analysis =====
#         if not self.is_training and getattr(self.cfg.evaluate, "prefix_rollout_ce", False):
#             tokens = audio_tokens.permute(0, 2, 1)   # (B, K, T) -> (B, T, K)
#             do_temporal = getattr(self.cfg.evaluate, "temporal_analysis", False)
#             window_sizes_sec = list(getattr(self.cfg.evaluate, "window_sizes_sec", [10, 20, 30]))

#             if do_temporal:
#                 rollout_ce, _, _ = self.model.prefix_rollout_ce(
#                     tokens,
#                     condition_tensors,
#                     prefix_ratio=self.cfg.evaluate.prefix_ratio,
#                     temporal_analysis=True,
#                     epoch=self.epoch,
#                     frame_rate=self.compression_model.frame_rate,
#                     window_sizes_sec=window_sizes_sec,
#                 )
#                 self.model.compute_normal_ce_temporal(
#                     tokens,
#                     condition_tensors,
#                     temporal_analysis=True,
#                     epoch=self.epoch,
#                     frame_rate=self.compression_model.frame_rate,
#                     window_sizes_sec=window_sizes_sec,
#                 )
#             else:
#                 rollout_ce = self.model.prefix_rollout_ce(
#                     tokens,
#                     condition_tensors,
#                     prefix_ratio=self.cfg.evaluate.prefix_ratio,
#                 )

#             metrics['prefix_rollout_ce'] = rollout_ce
#         # ===================================================
        
#         for k, ce_q in enumerate(ce_per_codebook):
#             metrics[f'ce_q{k + 1}'] = ce_q
#             metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

#         return metrics

#     @torch.no_grad()
#     def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
#                           gen_duration: float, prompt_duration: tp.Optional[float] = None,
#                           remove_text_conditioning: bool = False,
#                           ) -> dict:
#         """Run generate step on a batch of optional audio tensor and corresponding attributes.

#         Args:
#             batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
#             use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
#             gen_duration (float): Target audio duration for the generation.
#             prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
#         Returns:
#             gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
#                 and the prompt along with additional information.
#         """
#         bench_start = time.time()
#         audio, meta = batch
#         assert audio.size(0) == len(meta), (
#             f"Mismatch between number of items in audio batch ({audio.size(0)})",
#             f" and in metadata ({len(meta)})"
#         )
#         # prepare attributes
#         attributes = [x.to_condition_attributes() for x in meta]
#         if remove_text_conditioning:
#             attributes = _drop_description_condition(attributes)

#         # prepare audio prompt
#         if prompt_duration is None:
#             prompt_audio = None
#         else:
#             assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
#             prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
#             prompt_audio = audio[..., :prompt_audio_frames]

#         # get audio tokens from compression model
#         if prompt_audio is None or prompt_audio.nelement() == 0:
#             num_samples = len(attributes)
#             prompt_tokens = None
#         else:
#             num_samples = None
#             prompt_audio = prompt_audio.to(self.device)
#             prompt_tokens, scale = self.compression_model.encode(prompt_audio)
#             assert scale is None, "Compression model in MusicGen should not require rescaling."

#         # generate by sampling from the LM
#         with self.autocast:
#             total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
#             gen_tokens = self.model.generate(
#                 prompt_tokens, attributes, max_gen_len=total_gen_len,
#                 num_samples=num_samples, **self.generation_params)

#         # generate audio from tokens
#         assert gen_tokens.dim() == 3
#         gen_audio = self.compression_model.decode(gen_tokens, None)

#         bench_end = time.time()
#         gen_outputs = {
#             'rtf': (bench_end - bench_start) / gen_duration,
#             'ref_audio': audio,
#             'gen_audio': gen_audio,
#             'gen_tokens': gen_tokens,
#             'prompt_audio': prompt_audio,
#             'prompt_tokens': prompt_tokens,
#         }
#         return gen_outputs

#     def generate_audio(self) -> dict:
#         """Audio generation stage."""
#         generate_stage_name = f'{self.current_stage}'
#         sample_manager = SampleManager(self.xp)
#         self.logger.info(f"Generating samples in {sample_manager.base_folder}")
#         loader = self.dataloaders['generate']
#         updates = len(loader)
#         lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

#         dataset = get_dataset_from_loader(loader)
#         dataset_duration = dataset.segment_duration
#         assert dataset_duration is not None
#         assert isinstance(dataset, AudioDataset)
#         target_duration = self.cfg.generate.lm.gen_duration
#         prompt_duration = self.cfg.generate.lm.prompt_duration
#         if target_duration is None:
#             target_duration = dataset_duration
#         if prompt_duration is None:
#             prompt_duration = dataset_duration / 4
#         assert prompt_duration < dataset_duration, (
#             f"Specified prompt duration ({prompt_duration}s) is longer",
#             f" than reference audio duration ({dataset_duration}s)"
#         )

#         def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
#             hydrated_conditions = []
#             for sample in [x.to_condition_attributes() for x in meta]:
#                 cond_dict = {}
#                 for cond_type in sample.__annotations__.keys():
#                     for cond_key, cond_val in getattr(sample, cond_type).items():
#                         if cond_key not in self.model.condition_provider.conditioners.keys():
#                             continue
#                         if is_jsonable(cond_val):
#                             cond_dict[cond_key] = cond_val
#                         elif isinstance(cond_val, WavCondition):
#                             cond_dict[cond_key] = cond_val.path
#                         elif isinstance(cond_val, JointEmbedCondition):
#                             cond_dict[cond_key] = cond_val.text  # only support text at inference for now
#                         else:
#                             # if we reached this point, it is not clear how to log the condition
#                             # so we just log the type.
#                             cond_dict[cond_key] = str(type(cond_val))
#                             continue
#                 hydrated_conditions.append(cond_dict)
#             return hydrated_conditions

#         metrics: dict = {}
#         average = flashy.averager()
#         for batch in lp:
#             audio, meta = batch
#             # metadata for sample manager
#             hydrated_conditions = get_hydrated_conditions(meta)
#             sample_generation_params = {
#                 **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
#                 **self.generation_params
#             }
#             if self.cfg.generate.lm.unprompted_samples:
#                 if self.cfg.generate.lm.gen_gt_samples:
#                     # get the ground truth instead of generation
#                     self.logger.warn(
#                         "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
#                     gen_unprompted_audio = audio
#                     rtf = 1.
#                 else:
#                     gen_unprompted_outputs = self.run_generate_step(
#                         batch, gen_duration=target_duration, prompt_duration=None)
#                     gen_unprompted_audio = gen_unprompted_outputs['gen_audio'].cpu()
#                     rtf = gen_unprompted_outputs['rtf']
#                 sample_manager.add_samples(
#                     gen_unprompted_audio, self.epoch, hydrated_conditions,
#                     ground_truth_wavs=audio, generation_args=sample_generation_params)

#             if self.cfg.generate.lm.prompted_samples:
#                 gen_outputs = self.run_generate_step(
#                     batch, gen_duration=target_duration, prompt_duration=prompt_duration)
#                 gen_audio = gen_outputs['gen_audio'].cpu()
#                 prompt_audio = gen_outputs['prompt_audio'].cpu()
#                 sample_manager.add_samples(
#                     gen_audio, self.epoch, hydrated_conditions,
#                     prompt_wavs=prompt_audio, ground_truth_wavs=audio,
#                     generation_args=sample_generation_params)
#             if self.cfg.generate.lm.no_text_conditioning:
#                 gen_outputs = self.run_generate_step(
#                     batch, gen_duration=target_duration, prompt_duration=None,
#                     remove_text_conditioning=self.cfg.generate.lm.no_text_conditioning)
#                 gen_audio = gen_outputs['gen_audio'].cpu()
#                 rtf = gen_outputs['rtf']
#                 # Here, the prompt is the original audio provided for the style conditioning
#                 prompt_audio = gen_outputs['ref_audio'].cpu()
#                 sample_manager.add_samples(
#                     gen_audio, self.epoch, hydrated_conditions,
#                     prompt_wavs=prompt_audio, ground_truth_wavs=audio,
#                     generation_args=sample_generation_params)

#             metrics['rtf'] = rtf
#             metrics = average(metrics)

#         flashy.distrib.barrier()
#         return metrics

#     def generate(self) -> dict:
#         """Generate stage."""
#         self.model.eval()
#         with torch.no_grad():
#             return self.generate_audio()

#     def run_epoch(self):
#         if self.cfg.cache.write:
#             if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
#                 return
#         super().run_epoch()

#     def train(self):
#         """Train stage.
#         """
#         if self._cached_batch_writer is not None:
#             self._cached_batch_writer.start_epoch(self.epoch)
#         if self._cached_batch_loader is None:
#             dataset = get_dataset_from_loader(self.dataloaders['train'])
#             assert isinstance(dataset, AudioDataset)
#             dataset.current_epoch = self.epoch
#         else:
#             self._cached_batch_loader.start_epoch(self.epoch)
#         return super().train()


#     # ------------------------------------------------------------------ #
#     #  Temporal window helpers: beat tracking + windowed FAD              #
#     # ------------------------------------------------------------------ #

#     def _compute_beat_stats(
#         self,
#         waveform: torch.Tensor,   # (C, T) or (T,) float32 CPU
#         sample_rate: int,
#         window_sizes_sec: tp.List[int],
#     ) -> tp.Dict[str, float]:
#         """Beat-track a single waveform and return tempo + beat regularity
#         (std of inter-beat intervals) for each temporal window.

#         Returns dict with keys like 'beat_tempo_10s', 'beat_regularity_10s', etc.
#         """
#         # mix to mono, convert to numpy
#         wav_np = waveform.mean(0).numpy() if waveform.dim() == 2 else waveform.numpy()
#         wav_np = wav_np.astype(np.float32)

#         # full-sequence beat tracking
#         try:
#             tempo, beat_frames = librosa.beat.beat_track(y=wav_np, sr=sample_rate, units='time')
#             beat_times = np.array(beat_frames)  # already in seconds when units='time'
#         except Exception:
#             beat_times = np.array([])
#             tempo = 0.0

#         stats: tp.Dict[str, float] = {}
#         for w_sec in window_sizes_sec:
#             key = f'{w_sec}s'
#             window_beats = beat_times[beat_times <= w_sec]
#             if len(window_beats) >= 2:
#                 ibi = np.diff(window_beats)           # inter-beat intervals
#                 stats[f'beat_tempo_{key}'] = float(60.0 / ibi.mean()) if ibi.mean() > 0 else 0.0
#                 stats[f'beat_regularity_{key}'] = float(ibi.std())
#             else:
#                 stats[f'beat_tempo_{key}'] = 0.0
#                 stats[f'beat_regularity_{key}'] = 0.0
#         return stats

#     # def _compute_windowed_fad_beat(
#     #     self,
#     #     gen_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors, full length
#     #     ref_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors, full length
#     #     sample_rate: int,
#     #     window_sizes_sec: tp.List[int],
#     #     epoch: int,
#     # ) -> tp.Dict[str, float]:
#     #     """Compute FAD and beat stats for each temporal window slice.

#     #     For each window [0, w_sec]:
#     #       - Slice generated and reference audio to w_sec
#     #       - Compute FAD between gen slice distribution and ref slice distribution
#     #       - Compute beat tracking stats on generated slices

#     #     Results are also saved to $AUDIOCRAFT_DORA_DIR/temporal_logs/windowed_audio_metrics.jsonl
#     #     """
#     #     metrics: tp.Dict[str, float] = {}
#     #     all_beat_stats: tp.List[tp.Dict] = []

#     #     for w_sec in window_sizes_sec:
#     #         w_samples = int(w_sec * sample_rate)
#     #         key = f'{w_sec}s'

#     #         # ---- slice audio to window ----
#     #         gen_slices = [w[:, :w_samples] if w.dim() == 2 else w[:w_samples]
#     #                       for w in gen_wavs]
#     #         ref_slices = [w[:, :w_samples] if w.dim() == 2 else w[:w_samples]
#     #                       for w in ref_wavs]

#     #         # ---- windowed FAD ----
#     #         try:
#     #             fad_w = builders.get_fad(self.cfg.metrics.fad).to(self.device)
#     #             sizes_gen = torch.tensor([s.shape[-1] for s in gen_slices])
#     #             sizes_ref = torch.tensor([s.shape[-1] for s in ref_slices])
#     #             sr_tensor = torch.tensor([sample_rate] * len(gen_slices))
#     #             stems = [f'window_{key}_sample{i}' for i in range(len(gen_slices))]

#     #             gen_batch = torch.stack(gen_slices).unsqueeze(1) if gen_slices[0].dim() == 1                     else torch.stack(gen_slices)
#     #             ref_batch = torch.stack(ref_slices).unsqueeze(1) if ref_slices[0].dim() == 1                     else torch.stack(ref_slices)

#     #             fad_w.update(gen_batch, ref_batch, sizes_gen, sr_tensor, stems)
#     #             fad_val = float(fad_w.compute())
#     #             metrics[f'fad_{key}'] = fad_val
#     #         except Exception as e:
#     #             self.logger.warning(f"Windowed FAD failed for window {key}: {e}")
#     #             fad_val = -1.0
#     #             metrics[f'fad_{key}'] = fad_val

#     #         # ---- beat tracking on generated slices ----
#     #         window_beat_stats: tp.Dict[str, float] = {
#     #             f'beat_tempo_{key}': 0.0,
#     #             f'beat_regularity_{key}': 0.0,
#     #         }
#     #         beat_entries = []
#     #         for wav in gen_slices:
#     #             s = self._compute_beat_stats(wav, sample_rate, [w_sec])
#     #             beat_entries.append(s)
#     #         if beat_entries:
#     #             for stat_key in beat_entries[0]:
#     #                 window_beat_stats[stat_key] = float(np.mean([e[stat_key] for e in beat_entries]))
#     #         metrics.update(window_beat_stats)

#     #         all_beat_stats.append({
#     #             'window': key,
#     #             'fad': fad_val,
#     #             **window_beat_stats,
#     #         })

#     #     # ---- save to jsonl ----
#     #     dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
#     #     out_dir = os.path.join(dora_dir, "temporal_logs") if dora_dir else "temporal_logs"
#     #     os.makedirs(out_dir, exist_ok=True)
#     #     record = {'epoch': epoch, 'windows': all_beat_stats}
#     #     with open(os.path.join(out_dir, "windowed_audio_metrics.jsonl"), "a") as f:
#     #         f.write(json.dumps(record) + "\n")

#     #     return metrics
    
#     def _compute_windowed_fad_beat(
#         self,
#         gen_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
#         ref_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
#         sample_rate: int,
#         window_sizes_sec: tp.List[int],
#         epoch: int,
#     ) -> tp.Dict[str, float]:
#         """Compute FAD and beat stats for isolated, non-overlapping temporal windows."""
#         metrics: tp.Dict[str, float] = {}
#         all_beat_stats: tp.List[tp.Dict] = []

#         prev_sec = 0
#         prev_samples = 0

#         # Ensure chronological order
#         for w_sec in sorted(window_sizes_sec):
#             w_samples = int(w_sec * sample_rate)
#             key = f'{prev_sec}-{w_sec}s'
#             chunk_duration = w_sec - prev_sec

#             # ---- slice audio to ISOLATED window ----
#             gen_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
#                           for w in gen_wavs]
#             ref_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
#                           for w in ref_wavs]

#             # Skip if the slice is empty (e.g., generated audio is shorter than expected)
#             if any(s.shape[-1] == 0 for s in gen_slices):
#                 continue

#             # ---- windowed FAD ----
#             try:
#                 fad_w = builders.get_fad(self.cfg.metrics.fad).to(self.device)
#                 sizes_gen = torch.tensor([s.shape[-1] for s in gen_slices])
#                 sizes_ref = torch.tensor([s.shape[-1] for s in ref_slices])
#                 sr_tensor = torch.tensor([sample_rate] * len(gen_slices))
#                 stems = [f'window_{key}_sample{i}' for i in range(len(gen_slices))]

#                 gen_batch = torch.stack(gen_slices).unsqueeze(1) if gen_slices[0].dim() == 1 else torch.stack(gen_slices)
#                 ref_batch = torch.stack(ref_slices).unsqueeze(1) if ref_slices[0].dim() == 1 else torch.stack(ref_slices)

#                 fad_w.update(gen_batch, ref_batch, sizes_gen, sr_tensor, stems)
#                 fad_val = float(fad_w.compute())
#                 metrics[f'fad_{key}'] = fad_val
#             except Exception as e:
#                 self.logger.warning(f"Windowed FAD failed for window {key}: {e}")
#                 fad_val = -1.0
#                 metrics[f'fad_{key}'] = fad_val

#             # ---- beat tracking on generated slices ----
#             window_beat_stats: tp.Dict[str, float] = {
#                 f'beat_tempo_{key}': 0.0,
#                 f'beat_regularity_{key}': 0.0,
#             }
#             beat_entries = []
            
#             for wav in gen_slices:
#                 # We tell the beat tracker this is a standalone chunk of length `chunk_duration`
#                 s = self._compute_beat_stats(wav, sample_rate, [chunk_duration])
                
#                 # Remap the returned keys (e.g., '10s') to the target chunk key (e.g., '20-30s')
#                 adjusted_s = {
#                     f'beat_tempo_{key}': s.get(f'beat_tempo_{chunk_duration}s', 0.0),
#                     f'beat_regularity_{key}': s.get(f'beat_regularity_{chunk_duration}s', 0.0),
#                 }
#                 beat_entries.append(adjusted_s)

#             if beat_entries:
#                 for stat_key in window_beat_stats.keys():
#                     window_beat_stats[stat_key] = float(np.mean([e[stat_key] for e in beat_entries]))
#             metrics.update(window_beat_stats)

#             all_beat_stats.append({
#                 'window': key,
#                 'fad': fad_val,
#                 **window_beat_stats,
#             })

#             # Advance the boundaries for the next iteration
#             prev_samples = w_samples
#             prev_sec = w_sec

#         # ---- save to jsonl ----
#         dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
#         out_dir = os.path.join(dora_dir, "temporal_logs") if dora_dir else "temporal_logs"
#         os.makedirs(out_dir, exist_ok=True)
#         record = {'epoch': epoch, 'windows': all_beat_stats}
#         with open(os.path.join(out_dir, "windowed_audio_metrics.jsonl"), "a") as f:
#             f.write(json.dumps(record) + "\n")

#         return metrics

#     def evaluate_audio_generation(self) -> dict:
#         """Evaluate audio generation with off-the-shelf metrics."""
#         evaluate_stage_name = f'{self.current_stage}_generation'
#         # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
#         fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
#         kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
#         text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
#         chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
#         should_run_eval = False
#         eval_chroma_wavs: tp.Optional[torch.Tensor] = None
#         if self.cfg.evaluate.metrics.fad:
#             fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
#             should_run_eval = True
#         if self.cfg.evaluate.metrics.kld:
#             kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
#             should_run_eval = True
#         if self.cfg.evaluate.metrics.text_consistency:
#             text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
#             should_run_eval = True
#         if self.cfg.evaluate.metrics.chroma_cosine:
#             chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
#             # if we have predefind wavs for chroma we should purge them for computing the cosine metric
#             has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
#                                           self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
#             if has_predefined_eval_chromas:
#                 warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
#                                        'Resetting eval chromas to None for evaluation.')
#                 eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
#                 self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
#             should_run_eval = True

#         def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
#             audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
#             compressed_audio = self.compression_model.decode(audio_tokens, scale)
#             return compressed_audio[..., :audio.shape[-1]]

#         metrics: dict = {}
#         if should_run_eval:
#             loader = self.dataloaders['evaluate']
#             updates = len(loader)
#             lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
#             average = flashy.averager()
#             dataset = get_dataset_from_loader(loader)
#             assert isinstance(dataset, AudioDataset)
#             self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

#             # accumulators for windowed FAD + beat tracking
#             do_windowed = getattr(self.cfg.evaluate, "windowed_audio_metrics", False)
#             window_sizes_sec = list(getattr(self.cfg.evaluate, "window_sizes_sec", [10, 20, 30]))
#             all_gen_wavs: tp.List[torch.Tensor] = []
#             all_ref_wavs: tp.List[torch.Tensor] = []

#             for idx, batch in enumerate(lp):
#                 audio, meta = batch
#                 assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

#                 target_duration = audio.shape[-1] / self.cfg.sample_rate
#                 if self.cfg.evaluate.fixed_generation_duration:
#                     target_duration = self.cfg.evaluate.fixed_generation_duration

#                 gen_outputs = self.run_generate_step(
#                     batch, gen_duration=target_duration,
#                     remove_text_conditioning=self.cfg.evaluate.get('remove_text_conditioning', False)
#                 )
#                 y_pred = gen_outputs['gen_audio'].detach()
#                 y_pred = y_pred[..., :audio.shape[-1]]

#                 normalize_kwargs = dict(self.cfg.generate.audio)
#                 normalize_kwargs.pop('format', None)
#                 y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
#                 y = audio.cpu()  # should already be on CPU but just in case
#                 sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
#                 sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
#                 audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

#                 if fad is not None:
#                     if self.cfg.metrics.fad.use_gt:
#                         y_pred = get_compressed_audio(y).cpu()
#                     fad.update(y_pred, y, sizes, sample_rates, audio_stems)
#                 if kldiv is not None:
#                     if self.cfg.metrics.kld.use_gt:
#                         y_pred = get_compressed_audio(y).cpu()
#                     kldiv.update(y_pred, y, sizes, sample_rates)
#                 if text_consistency is not None:
#                     texts = [m.description for m in meta]
#                     if self.cfg.metrics.text_consistency.use_gt:
#                         y_pred = y
#                     text_consistency.update(y_pred, texts, sizes, sample_rates)
#                 if chroma_cosine is not None:
#                     if self.cfg.metrics.chroma_cosine.use_gt:
#                         y_pred = get_compressed_audio(y).cpu()
#                     chroma_cosine.update(y_pred, y, sizes, sample_rates)
#                     # restore chroma conditioner's eval chroma wavs
#                     if eval_chroma_wavs is not None:
#                         self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

#                 # accumulate for windowed metrics (collect individual samples, not batches)
#                 if do_windowed:
#                     for i in range(y_pred.shape[0]):
#                         all_gen_wavs.append(y_pred[i].cpu())
#                         all_ref_wavs.append(y[i].cpu())

#             flashy.distrib.barrier()

#             # ---- windowed FAD + beat tracking ----
#             if do_windowed and len(all_gen_wavs) > 0:
#                 windowed_metrics = self._compute_windowed_fad_beat(
#                     gen_wavs=all_gen_wavs,
#                     ref_wavs=all_ref_wavs,
#                     sample_rate=self.cfg.sample_rate,
#                     window_sizes_sec=window_sizes_sec,
#                     epoch=self.epoch,
#                 )
#                 metrics.update(windowed_metrics)

#             if fad is not None:
#                 metrics['fad'] = fad.compute()
#             if kldiv is not None:
#                 kld_metrics = kldiv.compute()
#                 metrics.update(kld_metrics)
#             if text_consistency is not None:
#                 metrics['text_consistency'] = text_consistency.compute()
#             if chroma_cosine is not None:
#                 metrics['chroma_cosine'] = chroma_cosine.compute()
                
#             metrics = average(metrics)
#             metrics = flashy.distrib.average_metrics(metrics, len(loader))

#         return metrics

#     def evaluate(self) -> dict:
#         """Evaluate stage."""
#         self.model.eval()
#         with torch.no_grad():
#             metrics: dict = {}
#             if self.cfg.evaluate.metrics.base:
#                 metrics.update(self.common_train_valid('evaluate'))
#             gen_metrics = self.evaluate_audio_generation()
#             return {**metrics, **gen_metrics}


# # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import json
import os
import time
import typing as tp
import warnings

import flashy
import librosa
import math
import numpy as np
import omegaconf
import torch
from torch.nn import functional as F

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, \
            StyleConditioner, _drop_description_condition
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash


class MusicGenSolver(base.StandardSolver):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # easier access to sampling parameters
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
        }
        self._best_metric_name: tp.Optional[str] = 'ce'

        self._cached_batch_writer = None
        self._cached_batch_loader = None
        if cfg.cache.path:
            if cfg.cache.write:
                self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
                if self.cfg.cache.write_num_shards:
                    self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
                    self._best_metric_name = None
            else:
                self._cached_batch_loader = CachedBatchLoader(
                    Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
                    min_length=self.cfg.optim.updates_per_epoch or 1)
                self.dataloaders['original_train'] = self.dataloaders['train']
                self.dataloaders['train'] = self._cached_batch_loader  # type: ignore

    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        """Mostly a convenience function around magma.train.get_solver_from_sig,
        populating all the proper param, deactivating EMA, FSDP, loading the best state,
        basically all you need to get a solver ready to "play" with in single GPU mode
        and with minimal memory overhead.

        Args:
            sig (str): signature to load.
            dtype (str or None): potential dtype, as a string, i.e. 'float16'.
            device (str or None): potential device, as a string, i.e. 'cuda'.
            override_cfg (dict or omegaconf.DictConfig or None): potential device, as a string, i.e. 'cuda'.
        """
        from audiocraft import train
        our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
        our_override_cfg['autocast'] = autocast
        if dtype is not None:
            our_override_cfg['dtype'] = dtype
        if device is not None:
            our_override_cfg['device'] = device
        if batch_size is not None:
            our_override_cfg['dataset'] = {'batch_size': batch_size}
        if override_cfg is None:
            override_cfg = {}
        override_cfg = omegaconf.OmegaConf.merge(
            omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))  # type: ignore
        solver = train.get_solver_from_sig(
            sig, override_cfg=override_cfg,
            load_best=True, disable_fsdp=True,
            ignore_state_keys=['optimizer', 'ema'], **kwargs)
        solver.model.eval()
        return solver

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter({
            'lr': '.2E',
            'ce': '.3f',
            'ppl': '.3f',
            'grad_norm': '.3E',
        }, exclude_keys=['ce_q*', 'ppl_q*'])

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        return self._best_metric_name

    def initialize_optimization(self) -> None:
        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        self.register_ema('model')
        # initialize optimization
        self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg.autocast_dtype]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == 'float16'
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            self.register_stateful('scaler')

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
        assert self.compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}"
        )
        assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
        )
        self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
                         self.compression_model.num_codebooks, self.compression_model.cardinality,
                         self.compression_model.frame_rate)
        # instantiate LM model
        self.model: tp.Union[models.LMModel, models.FlowMatchingModel] = models.builders.get_lm_model(
            self.cfg).to(self.device)

        # initialize optimization
        self.initialize_optimization()

    def build_dataloaders(self) -> None:
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

    def show(self) -> None:
        """Show the compression model and LM model."""
        self.logger.info("Compression model:")
        self.log_model_summary(self.compression_model)
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    def load_state_dict(self, state: dict) -> None:
        if 'condition_provider' in state:
            model_state = state['model']
            condition_provider_state = state.pop('condition_provider')
            prefix = 'condition_provider.'
            for key, value in condition_provider_state.items():
                key = prefix + key
                assert key not in model_state
                model_state[key] = value
        if 'compression_model' in state:
            # We used to store the `compression_model` state in the checkpoint, however
            # this is in general not needed, as the compression model should always be readable
            # from the original `cfg.compression_model_checkpoint` location.
            compression_model_state = state.pop('compression_model')
            before_hash = model_hash(self.compression_model)
            self.compression_model.load_state_dict(compression_model_state)
            after_hash = model_hash(self.compression_model)
            if before_hash != after_hash:
                raise RuntimeError(
                    "The compression model state inside the checkpoint is different"
                    " from the one obtained from compression_model_checkpoint..."
                    "We do not support altering the compression model inside the LM "
                    "checkpoint as parts of the code, in particular for running eval post-training "
                    "will use the compression_model_checkpoint as the source of truth.")

        super().load_state_dict(state)

    def load_from_pretrained(self, name: str):
        # TODO: support native HF versions of MusicGen.
        lm_pkg = models.loaders.load_lm_model_ckpt(name)
        state: dict = {
            'best_state': {
                'model': lm_pkg['best_state'],
            },
        }
        return state

    # def _compute_cross_entropy(
    #     self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    # ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
    #     """Compute cross entropy between multi-codebook targets and model's logits.
    #     The cross entropy is computed per codebook to provide codebook-level cross entropy.
    #     Valid timesteps for each of the codebook are pulled from the mask, where invalid
    #     timesteps are set to 0.

    #     Args:
    #         logits (torch.Tensor): Model's logits of shape [B, K, T, card].
    #         targets (torch.Tensor): Target codes, of shape [B, K, T].
    #         mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
    #     Returns:
    #         ce (torch.Tensor): Cross entropy averaged over the codebooks
    #         ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
    #     """
    #     B, K, T = targets.shape
    #     assert logits.shape[:-1] == targets.shape
    #     assert mask.shape == targets.shape
    #     ce = torch.zeros([], device=targets.device)
    #     ce_per_codebook: tp.List[torch.Tensor] = []
    #     for k in range(K):
    #         logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
    #         targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
    #         mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
    #         ce_targets = targets_k[mask_k]
    #         ce_logits = logits_k[mask_k]
    #         q_ce = F.cross_entropy(ce_logits, ce_targets)
    #         ce += q_ce
    #         ce_per_codebook.append(q_ce.detach())
    #     # average cross entropy across codebooks
    #     ce = ce / K
    #     return ce, ce_per_codebook
    
    def _compute_cross_entropy(
            self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
        ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
            B, K, T = targets.shape
            assert logits.shape[:-1] == targets.shape
            assert mask.shape == targets.shape
            ce = torch.zeros([], device=targets.device, dtype=torch.float32)
            ce_per_codebook: tp.List[torch.Tensor] = []
            for k in range(K):
                logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
                targets_k = targets[:, k, ...].contiguous().view(-1)                 # [B x T]
                mask_k = mask[:, k, ...].contiguous().view(-1)                       # [B x T] bool
                ce_targets = targets_k[mask_k]
                ce_logits = logits_k[mask_k].float()  # boolean index excludes padding, float32 prevents fp16 overflow
                if ce_targets.numel() == 0:
                    continue
                q_ce = F.cross_entropy(ce_logits, ce_targets)
                ce += q_ce
                ce_per_codebook.append(q_ce.detach())
            ce = ce / K
            return ce, ce_per_codebook

    def _get_audio_tokens(self, audio: torch.Tensor):
        with torch.no_grad():
            audio_tokens, scale = self.compression_model.encode(audio)
            assert scale is None, "Scaled compression model not supported with LM."
            return audio_tokens

    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
        check_synchronization_points: bool = False
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        if self.model.training:
            warnings.warn(
                "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
                "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
                "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
                "Really sorry about that.")
        if self._cached_batch_loader is None or self.current_stage != "train":
            audio, infos = batch
            audio = audio.to(self.device)
            audio_tokens = None
            assert audio.size(0) == len(infos), (
                f"Mismatch between number of items in audio batch ({audio.size(0)})",
                f" and in metadata ({len(infos)})"
            )
        else:
            audio = None
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            infos, = batch  # type: ignore
            assert all([isinstance(info, AudioInfo) for info in infos])
            assert all([info.audio_tokens is not None for info in infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            for info in infos:
                if isinstance(info, MusicInfo):
                    # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                    # then you must be using the chroma cache! otherwise the code will try
                    # to use this segment and fail (by that I mean you will see NaN everywhere).
                    info.self_wav = WavCondition(
                        torch.full([1, info.channels, info.total_frames], float('NaN')),
                        length=torch.tensor([info.n_frames]),
                        sample_rate=[info.sample_rate],
                        path=[info.meta.path],
                        seek_time=[info.seek_time])
                    dataset = get_dataset_from_loader(self.dataloaders['original_train'])
                    assert isinstance(dataset, MusicDataset), type(dataset)
                    if dataset.paraphraser is not None and info.description is not None:
                        # Hackingly reapplying paraphraser when using cache.
                        info.description = dataset.paraphraser.sample_paraphrase(
                            info.meta.path, info.description)
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            audio_tokens = self._get_audio_tokens(audio)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames
                audio_sample_rate = infos[i].sample_rate
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None and self.current_stage == 'train':
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for info, one_audio_tokens in zip(infos, audio_tokens):
                assert isinstance(info, AudioInfo)
                if isinstance(info, MusicInfo):
                    assert not info.joint_embed, "joint_embed and cache not supported yet."
                    info.self_wav = None
                assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
                info.audio_tokens = one_audio_tokens.short().cpu()
            self._cached_batch_writer.save(infos)

        return condition_tensors, audio_tokens, padding_mask

    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        with self.autocast:
            style_mask = None
            if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
                if isinstance(self.model.condition_provider.conditioners.self_wav, StyleConditioner):
                    style_mask = self.model.condition_provider.conditioners.self_wav.mask

            # ---- SASS: use structure-aware scheduled sampling if enabled ----
            use_sass = self.is_training and getattr(self.cfg, 'sass', None) is not None \
                       and getattr(self.cfg.sass, 'enabled', False)
            if use_sass:
                sass_cfg = self.cfg.sass
                steps_per_epoch = len(self.dataloaders['train'])
                current_step = (self.epoch - 1) * steps_per_epoch + idx
                total_steps = self.cfg.optim.epochs * steps_per_epoch
                model_output = self.model.compute_sass_predictions(
                    audio_tokens, [], condition_tensors,
                    current_step=current_step,
                    total_steps=total_steps,
                    p_max=getattr(sass_cfg, 'p_max', 0.20),
                    warmup_start=getattr(sass_cfg, 'warmup_start', 5000),
                    warmup_ramp=getattr(sass_cfg, 'warmup_ramp', 8000),
                    decay_ratio=getattr(sass_cfg, 'decay_ratio', 0.7),
                    alpha=getattr(sass_cfg, 'alpha', 1.0),
                    c_min=getattr(sass_cfg, 'c_min', 0.35),
                    c_max=getattr(sass_cfg, 'c_max', 0.85),
                    token_cap=getattr(sass_cfg, 'token_cap', 0.20),
                    codebook_weights=getattr(sass_cfg, 'codebook_weights', [0.3, 0.7, 1.0, 1.0]),
                    temp=getattr(sass_cfg, 'temp', 0.8),
                    top_k=getattr(sass_cfg, 'top_k', 50),
                )  # type: ignore

            else:
                model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            if style_mask is not None:
                mask = padding_mask & model_output.mask & style_mask
            else:
                mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce
        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update('scale')
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)
        
        # ===== Prefix rollout CE + temporal analysis =====
        if not self.is_training and getattr(self.cfg.evaluate, "prefix_rollout_ce", False):
            tokens = audio_tokens.permute(0, 2, 1)   # (B, K, T) -> (B, T, K)
            do_temporal = getattr(self.cfg.evaluate, "temporal_analysis", False)
            window_sizes_sec = list(getattr(self.cfg.evaluate, "window_sizes_sec", [10, 20, 30]))

            if do_temporal:
                rollout_ce, _, _ = self.model.prefix_rollout_ce(
                    tokens,
                    condition_tensors,
                    prefix_ratio=self.cfg.evaluate.prefix_ratio,
                    temporal_analysis=True,
                    epoch=self.epoch,
                    frame_rate=self.compression_model.frame_rate,
                    window_sizes_sec=window_sizes_sec,
                )
                self.model.compute_normal_ce_temporal(
                    tokens,
                    condition_tensors,
                    temporal_analysis=True,
                    epoch=self.epoch,
                    frame_rate=self.compression_model.frame_rate,
                    window_sizes_sec=window_sizes_sec,
                )
            else:
                rollout_ce = self.model.prefix_rollout_ce(
                    tokens,
                    condition_tensors,
                    prefix_ratio=self.cfg.evaluate.prefix_ratio,
                )

            metrics['prefix_rollout_ce'] = rollout_ce
        # ===================================================
        
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_text_conditioning: bool = False,
                          ) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        audio, meta = batch
        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]
        if remove_text_conditioning:
            attributes = _drop_description_condition(attributes)

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f'{self.current_stage}'
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)"
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            audio, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
                    gen_unprompted_audio = audio
                    rtf = 1.
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch, gen_duration=target_duration, prompt_duration=None)
                    gen_unprompted_audio = gen_unprompted_outputs['gen_audio'].cpu()
                    rtf = gen_unprompted_outputs['rtf']
                sample_manager.add_samples(
                    gen_unprompted_audio, self.epoch, hydrated_conditions,
                    ground_truth_wavs=audio, generation_args=sample_generation_params)

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=prompt_duration)
                gen_audio = gen_outputs['gen_audio'].cpu()
                prompt_audio = gen_outputs['prompt_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)
            if self.cfg.generate.lm.no_text_conditioning:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=None,
                    remove_text_conditioning=self.cfg.generate.lm.no_text_conditioning)
                gen_audio = gen_outputs['gen_audio'].cpu()
                rtf = gen_outputs['rtf']
                # Here, the prompt is the original audio provided for the style conditioning
                prompt_audio = gen_outputs['ref_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)

            metrics['rtf'] = rtf
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def generate(self) -> dict:
        """Generate stage."""
        self.model.eval()
        with torch.no_grad():
            return self.generate_audio()

    def run_epoch(self):
        if self.cfg.cache.write:
            if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
                return
        super().run_epoch()

    def train(self):
        """Train stage.
        """
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.epoch
        else:
            self._cached_batch_loader.start_epoch(self.epoch)
        return super().train()


    # ------------------------------------------------------------------ #
    #  Temporal window helpers: beat tracking + windowed FAD              #
    # ------------------------------------------------------------------ #

    def _compute_beat_stats(
        self,
        waveform: torch.Tensor,   # (C, T) or (T,) float32 CPU
        sample_rate: int,
        window_sizes_sec: tp.List[int],
    ) -> tp.Dict[str, float]:
        """Beat-track a single waveform and return tempo + beat regularity
        (std of inter-beat intervals) for each temporal window.

        Returns dict with keys like 'beat_tempo_10s', 'beat_regularity_10s', etc.
        """
        # mix to mono, convert to numpy
        wav_np = waveform.mean(0).numpy() if waveform.dim() == 2 else waveform.numpy()
        wav_np = wav_np.astype(np.float32)

        # full-sequence beat tracking
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=wav_np, sr=sample_rate, units='time')
            beat_times = np.array(beat_frames)  # already in seconds when units='time'
        except Exception:
            beat_times = np.array([])
            tempo = 0.0

        stats: tp.Dict[str, float] = {}
        for w_sec in window_sizes_sec:
            key = f'{w_sec}s'
            window_beats = beat_times[beat_times <= w_sec]
            if len(window_beats) >= 2:
                ibi = np.diff(window_beats)           # inter-beat intervals
                stats[f'beat_tempo_{key}'] = float(60.0 / ibi.mean()) if ibi.mean() > 0 else 0.0
                stats[f'beat_regularity_{key}'] = float(ibi.std())
            else:
                stats[f'beat_tempo_{key}'] = 0.0
                stats[f'beat_regularity_{key}'] = 0.0
        return stats

    # def _compute_windowed_fad_beat(
    #     self,
    #     gen_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors, full length
    #     ref_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors, full length
    #     sample_rate: int,
    #     window_sizes_sec: tp.List[int],
    #     epoch: int,
    # ) -> tp.Dict[str, float]:
    #     """Compute FAD and beat stats for each temporal window slice.

    #     For each window [0, w_sec]:
    #       - Slice generated and reference audio to w_sec
    #       - Compute FAD between gen slice distribution and ref slice distribution
    #       - Compute beat tracking stats on generated slices

    #     Results are also saved to $AUDIOCRAFT_DORA_DIR/temporal_logs/windowed_audio_metrics.jsonl
    #     """
    #     metrics: tp.Dict[str, float] = {}
    #     all_beat_stats: tp.List[tp.Dict] = []

    #     for w_sec in window_sizes_sec:
    #         w_samples = int(w_sec * sample_rate)
    #         key = f'{w_sec}s'

    #         # ---- slice audio to window ----
    #         gen_slices = [w[:, :w_samples] if w.dim() == 2 else w[:w_samples]
    #                       for w in gen_wavs]
    #         ref_slices = [w[:, :w_samples] if w.dim() == 2 else w[:w_samples]
    #                       for w in ref_wavs]

    #         # ---- windowed FAD ----
    #         try:
    #             fad_w = builders.get_fad(self.cfg.metrics.fad).to(self.device)
    #             sizes_gen = torch.tensor([s.shape[-1] for s in gen_slices])
    #             sizes_ref = torch.tensor([s.shape[-1] for s in ref_slices])
    #             sr_tensor = torch.tensor([sample_rate] * len(gen_slices))
    #             stems = [f'window_{key}_sample{i}' for i in range(len(gen_slices))]

    #             gen_batch = torch.stack(gen_slices).unsqueeze(1) if gen_slices[0].dim() == 1                     else torch.stack(gen_slices)
    #             ref_batch = torch.stack(ref_slices).unsqueeze(1) if ref_slices[0].dim() == 1                     else torch.stack(ref_slices)

    #             fad_w.update(gen_batch, ref_batch, sizes_gen, sr_tensor, stems)
    #             fad_val = float(fad_w.compute())
    #             metrics[f'fad_{key}'] = fad_val
    #         except Exception as e:
    #             self.logger.warning(f"Windowed FAD failed for window {key}: {e}")
    #             fad_val = -1.0
    #             metrics[f'fad_{key}'] = fad_val

    #         # ---- beat tracking on generated slices ----
    #         window_beat_stats: tp.Dict[str, float] = {
    #             f'beat_tempo_{key}': 0.0,
    #             f'beat_regularity_{key}': 0.0,
    #         }
    #         beat_entries = []
    #         for wav in gen_slices:
    #             s = self._compute_beat_stats(wav, sample_rate, [w_sec])
    #             beat_entries.append(s)
    #         if beat_entries:
    #             for stat_key in beat_entries[0]:
    #                 window_beat_stats[stat_key] = float(np.mean([e[stat_key] for e in beat_entries]))
    #         metrics.update(window_beat_stats)

    #         all_beat_stats.append({
    #             'window': key,
    #             'fad': fad_val,
    #             **window_beat_stats,
    #         })

    #     # ---- save to jsonl ----
    #     dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
    #     out_dir = os.path.join(dora_dir, "temporal_logs") if dora_dir else "temporal_logs"
    #     os.makedirs(out_dir, exist_ok=True)
    #     record = {'epoch': epoch, 'windows': all_beat_stats}
    #     with open(os.path.join(out_dir, "windowed_audio_metrics.jsonl"), "a") as f:
    #         f.write(json.dumps(record) + "\n")

    #     return metrics
    
    # def _compute_windowed_fad_beat(
    #     self,
    #     gen_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
    #     ref_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
    #     sample_rate: int,
    #     window_sizes_sec: tp.List[int],
    #     epoch: int,
    # ) -> tp.Dict[str, float]:
    #     """Compute FAD and beat stats for isolated, non-overlapping temporal windows."""
    #     metrics: tp.Dict[str, float] = {}
    #     all_beat_stats: tp.List[tp.Dict] = []

    #     prev_sec = 0
    #     prev_samples = 0

    #     # Ensure chronological order
    #     for w_sec in sorted(window_sizes_sec):
    #         w_samples = int(w_sec * sample_rate)
    #         key = f'{prev_sec}-{w_sec}s'
    #         chunk_duration = w_sec - prev_sec

    #         # ---- slice audio to ISOLATED window ----
    #         gen_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
    #                       for w in gen_wavs]
    #         ref_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
    #                       for w in ref_wavs]

    #         # Skip if the slice is empty (e.g., generated audio is shorter than expected)
    #         if any(s.shape[-1] == 0 for s in gen_slices):
    #             continue

    #         # ---- windowed FAD ----
    #         try:
    #             torch.cuda.empty_cache()
    #             fad_w = builders.get_fad(self.cfg.metrics.fad).to(self.device)
    #             sizes_gen = torch.tensor([s.shape[-1] for s in gen_slices])
    #             sizes_ref = torch.tensor([s.shape[-1] for s in ref_slices])
    #             sr_tensor = torch.tensor([sample_rate] * len(gen_slices))
    #             stems = [f'window_{key}_sample{i}' for i in range(len(gen_slices))]

    #             gen_batch = torch.stack(gen_slices).unsqueeze(1) if gen_slices[0].dim() == 1 else torch.stack(gen_slices)
    #             ref_batch = torch.stack(ref_slices).unsqueeze(1) if ref_slices[0].dim() == 1 else torch.stack(ref_slices)

    #             fad_w.update(gen_batch, ref_batch, sizes_gen, sr_tensor, stems)
    #             fad_val = float(fad_w.compute())
    #             metrics[f'fad_{key}'] = fad_val
    #         except Exception as e:
    #             self.logger.warning(f"Windowed FAD failed for window {key}: {e}")
    #             fad_val = -1.0
    #             metrics[f'fad_{key}'] = fad_val

    #         # ---- beat tracking on generated slices ----
    #         window_beat_stats: tp.Dict[str, float] = {
    #             f'beat_tempo_{key}': 0.0,
    #             f'beat_regularity_{key}': 0.0,
    #         }
    #         beat_entries = []
            
    #         for wav in gen_slices:
    #             # We tell the beat tracker this is a standalone chunk of length `chunk_duration`
    #             s = self._compute_beat_stats(wav, sample_rate, [chunk_duration])
                
    #             # Remap the returned keys (e.g., '10s') to the target chunk key (e.g., '20-30s')
    #             adjusted_s = {
    #                 f'beat_tempo_{key}': s.get(f'beat_tempo_{chunk_duration}s', 0.0),
    #                 f'beat_regularity_{key}': s.get(f'beat_regularity_{chunk_duration}s', 0.0),
    #             }
    #             beat_entries.append(adjusted_s)

    #         if beat_entries:
    #             for stat_key in window_beat_stats.keys():
    #                 window_beat_stats[stat_key] = float(np.mean([e[stat_key] for e in beat_entries]))
    #         metrics.update(window_beat_stats)

    #         all_beat_stats.append({
    #             'window': key,
    #             'fad': fad_val,
    #             **window_beat_stats,
    #         })

    #         # Advance the boundaries for the next iteration
    #         prev_samples = w_samples
    #         prev_sec = w_sec

    #     # ---- save to jsonl ----
    #     dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
    #     out_dir = os.path.join(dora_dir, "temporal_logs") if dora_dir else "temporal_logs"
    #     os.makedirs(out_dir, exist_ok=True)
    #     record = {'epoch': epoch, 'windows': all_beat_stats}
    #     with open(os.path.join(out_dir, "windowed_audio_metrics.jsonl"), "a") as f:
    #         f.write(json.dumps(record) + "\n")

    #     return metrics

    # def evaluate_audio_generation(self) -> dict:
    #     """Evaluate audio generation with off-the-shelf metrics."""
    #     evaluate_stage_name = f'{self.current_stage}_generation'
    #     # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
    #     fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
    #     kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
    #     text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
    #     chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
    #     should_run_eval = False
    #     eval_chroma_wavs: tp.Optional[torch.Tensor] = None
    #     if self.cfg.evaluate.metrics.fad:
    #         fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
    #         should_run_eval = True
    #     if self.cfg.evaluate.metrics.kld:
    #         kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
    #         should_run_eval = True
    #     if self.cfg.evaluate.metrics.text_consistency:
    #         text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
    #         should_run_eval = True
    #     if self.cfg.evaluate.metrics.chroma_cosine:
    #         chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
    #         # if we have predefind wavs for chroma we should purge them for computing the cosine metric
    #         has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
    #                                       self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
    #         if has_predefined_eval_chromas:
    #             warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
    #                                    'Resetting eval chromas to None for evaluation.')
    #             eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
    #             self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
    #         should_run_eval = True

    #     def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
    #         audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
    #         compressed_audio = self.compression_model.decode(audio_tokens, scale)
    #         return compressed_audio[..., :audio.shape[-1]]

    #     metrics: dict = {}
    #     if should_run_eval:
    #         loader = self.dataloaders['evaluate']
    #         updates = len(loader)
    #         lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
    #         average = flashy.averager()
    #         dataset = get_dataset_from_loader(loader)
    #         assert isinstance(dataset, AudioDataset)
    #         self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

    #         # accumulators for windowed FAD + beat tracking
    #         do_windowed = getattr(self.cfg.evaluate, "windowed_audio_metrics", False)
    #         window_sizes_sec = list(getattr(self.cfg.evaluate, "window_sizes_sec", [10, 20, 30]))
    #         all_gen_wavs: tp.List[torch.Tensor] = []
    #         all_ref_wavs: tp.List[torch.Tensor] = []

    #         for idx, batch in enumerate(lp):
    #             audio, meta = batch
    #             assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

    #             target_duration = audio.shape[-1] / self.cfg.sample_rate
    #             if self.cfg.evaluate.fixed_generation_duration:
    #                 target_duration = self.cfg.evaluate.fixed_generation_duration

    #             gen_outputs = self.run_generate_step(
    #                 batch, gen_duration=target_duration,
    #                 remove_text_conditioning=self.cfg.evaluate.get('remove_text_conditioning', False)
    #             )
    #             y_pred = gen_outputs['gen_audio'].detach()
    #             y_pred = y_pred[..., :audio.shape[-1]]

    #             normalize_kwargs = dict(self.cfg.generate.audio)
    #             normalize_kwargs.pop('format', None)
    #             y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
    #             y = audio.cpu()  # should already be on CPU but just in case
    #             sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
    #             sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
    #             audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

    #             if fad is not None:
    #                 if self.cfg.metrics.fad.use_gt:
    #                     y_pred = get_compressed_audio(y).cpu()
    #                 fad.update(y_pred, y, sizes, sample_rates, audio_stems)
    #             if kldiv is not None:
    #                 if self.cfg.metrics.kld.use_gt:
    #                     y_pred = get_compressed_audio(y).cpu()
    #                 kldiv.update(y_pred, y, sizes, sample_rates)
    #             if text_consistency is not None:
    #                 texts = [m.description for m in meta]
    #                 if self.cfg.metrics.text_consistency.use_gt:
    #                     y_pred = y
    #                 text_consistency.update(y_pred, texts, sizes, sample_rates)
    #             if chroma_cosine is not None:
    #                 if self.cfg.metrics.chroma_cosine.use_gt:
    #                     y_pred = get_compressed_audio(y).cpu()
    #                 chroma_cosine.update(y_pred, y, sizes, sample_rates)
    #                 # restore chroma conditioner's eval chroma wavs
    #                 if eval_chroma_wavs is not None:
    #                     self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

    #             # accumulate for windowed metrics (collect individual samples, not batches)
    #             if do_windowed:
    #                 for i in range(y_pred.shape[0]):
    #                     all_gen_wavs.append(y_pred[i].cpu())
    #                     all_ref_wavs.append(y[i].cpu())

    #         flashy.distrib.barrier()

    #         # ---- windowed FAD + beat tracking ----
    #         if do_windowed and len(all_gen_wavs) > 0:
    #             windowed_metrics = self._compute_windowed_fad_beat(
    #                 gen_wavs=all_gen_wavs,
    #                 ref_wavs=all_ref_wavs,
    #                 sample_rate=self.cfg.sample_rate,
    #                 window_sizes_sec=window_sizes_sec,
    #                 epoch=self.epoch,
    #             )
    #             metrics.update(windowed_metrics)

    #         if fad is not None:
    #             metrics['fad'] = fad.compute()
    #         if kldiv is not None:
    #             kld_metrics = kldiv.compute()
    #             metrics.update(kld_metrics)
    #         if text_consistency is not None:
    #             metrics['text_consistency'] = text_consistency.compute()
    #         if chroma_cosine is not None:
    #             metrics['chroma_cosine'] = chroma_cosine.compute()
                
    #         metrics = average(metrics)
    #         metrics = flashy.distrib.average_metrics(metrics, len(loader))

    #     return metrics

    def _compute_windowed_fad_beat(
        self,
        gen_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
        ref_wavs: tp.List[torch.Tensor],   # list of (C, T) CPU tensors
        sample_rate: int,
        window_sizes_sec: tp.List[int],
        epoch: int,
    ) -> tp.Dict[str, float]:
        """Compute FAD and beat stats for isolated, non-overlapping temporal windows.
 
        Collective-safe wrapper: gathers wavs from every rank, runs the actual
        computation on rank 0 only, then broadcasts the metrics dict so every
        rank returns the same result and no collective mismatches can happen.
 
        This prevents the epoch-10 crash caused by:
          1. `len(all_gen_wavs) > 0` gate diverging across ranks.
          2. `continue` on empty slices firing on some ranks but not others.
          3. torchmetrics `.compute()` all-reduces inside a per-window loop
             that different ranks traverse a different number of times.
        """
        # Gather all ranks' wavs on rank 0 (list-of-lists of CPU tensors).
        # flashy 0.0.2 doesn't expose all_gather_object, so use torch.distributed
        # directly. Fall back to single-process mode if distributed isn't initialized.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_gen: tp.List[tp.List[torch.Tensor]] = [None] * world_size  # type: ignore
            gathered_ref: tp.List[tp.List[torch.Tensor]] = [None] * world_size  # type: ignore
            # gather_object only populates the dst rank's output list.
            dist.gather_object(
                gen_wavs,
                gathered_gen if rank == 0 else None,
                dst=0,
            )
            dist.gather_object(
                ref_wavs,
                gathered_ref if rank == 0 else None,
                dst=0,
            )
        else:
            rank = 0
            gathered_gen = [gen_wavs]
            gathered_ref = [ref_wavs]
 
        if rank == 0:
            all_gen: tp.List[torch.Tensor] = [w for sub in gathered_gen for w in sub]
            all_ref: tp.List[torch.Tensor] = [w for sub in gathered_ref for w in sub]
            metrics: tp.Optional[tp.Dict[str, float]] = self._compute_windowed_fad_beat_local(
                all_gen, all_ref, sample_rate, window_sizes_sec, epoch,
            )
        else:
            metrics = None
 
        metrics = flashy.distrib.broadcast_object(metrics, src=0)
        return metrics or {}
 
    def _compute_windowed_fad_beat_local(
        self,
        gen_wavs: tp.List[torch.Tensor],
        ref_wavs: tp.List[torch.Tensor],
        sample_rate: int,
        window_sizes_sec: tp.List[int],
        epoch: int,
    ) -> tp.Dict[str, float]:
        """Rank-0-only implementation. Must NOT call any torch.distributed
        collective, because non-zero ranks are parked in the outer
        broadcast_object. In particular, do NOT call torchmetrics
        `fad_w.compute()` here — it performs a dist all-reduce on its state.
        Use the private `_local_compute_frechet_audio_distance()` instead,
        which runs the FAD subprocesses without any sync."""
        metrics: tp.Dict[str, float] = {}
        all_beat_stats: tp.List[tp.Dict] = []
 
        prev_sec = 0
        prev_samples = 0
 
        # Ensure chronological order
        for w_sec in sorted(window_sizes_sec):
            w_samples = int(w_sec * sample_rate)
            key = f'{prev_sec}-{w_sec}s'
            chunk_duration = w_sec - prev_sec
 
            # ---- slice audio to ISOLATED window ----
            gen_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
                          for w in gen_wavs]
            ref_slices = [w[:, prev_samples:w_samples] if w.dim() == 2 else w[prev_samples:w_samples]
                          for w in ref_wavs]
 
            # Skip if the slice is empty. Safe here: rank 0 only, no peer waiting.
            if not gen_slices or any(s.shape[-1] == 0 for s in gen_slices):
                prev_samples = w_samples
                prev_sec = w_sec
                continue
 
            # ---- windowed FAD ----
            fad_val = -1.0
            try:
                # torch.cuda.empty_cache()
                fad_w = builders.get_fad(self.cfg.metrics.fad).to(self.device)
                sizes_gen = torch.tensor([s.shape[-1] for s in gen_slices])
                sizes_ref = torch.tensor([s.shape[-1] for s in ref_slices])
                sr_tensor = torch.tensor([sample_rate] * len(gen_slices))
                stems = [f'window_{key}_sample{i}' for i in range(len(gen_slices))]
 
                gen_batch = torch.stack(gen_slices).unsqueeze(1) if gen_slices[0].dim() == 1 else torch.stack(gen_slices)
                ref_batch = torch.stack(ref_slices).unsqueeze(1) if ref_slices[0].dim() == 1 else torch.stack(ref_slices)
 
                fad_w.update(gen_batch, ref_batch, sizes_gen, sr_tensor, stems)
 
                # IMPORTANT: use the private local compute to avoid torchmetrics
                # _sync_dist, which would all_reduce on `total_files` and deadlock
                # because non-zero ranks are sitting in broadcast_object.
                fad_score = fad_w._local_compute_frechet_audio_distance()
                fad_val = float(fad_score) if fad_score is not None else -1.0
                metrics[f'fad_{key}'] = fad_val
            except Exception as e:
                self.logger.warning(f"Windowed FAD failed for window {key}: {e}")
                fad_val = -1.0
                metrics[f'fad_{key}'] = fad_val
 
            # ---- beat tracking on generated slices ----
            window_beat_stats: tp.Dict[str, float] = {
                f'beat_tempo_{key}': 0.0,
                f'beat_regularity_{key}': 0.0,
            }
            beat_entries = []
 
            for wav in gen_slices:
                # We tell the beat tracker this is a standalone chunk of length `chunk_duration`
                s = self._compute_beat_stats(wav, sample_rate, [chunk_duration])
 
                # Remap the returned keys (e.g., '10s') to the target chunk key (e.g., '20-30s')
                adjusted_s = {
                    f'beat_tempo_{key}': s.get(f'beat_tempo_{chunk_duration}s', 0.0),
                    f'beat_regularity_{key}': s.get(f'beat_regularity_{chunk_duration}s', 0.0),
                }
                beat_entries.append(adjusted_s)
 
            if beat_entries:
                for stat_key in window_beat_stats.keys():
                    window_beat_stats[stat_key] = float(np.mean([e[stat_key] for e in beat_entries]))
            metrics.update(window_beat_stats)
 
            all_beat_stats.append({
                'window': key,
                'fad': fad_val,
                **window_beat_stats,
            })
 
            # Advance the boundaries for the next iteration
            prev_samples = w_samples
            prev_sec = w_sec
 
        # ---- save to jsonl (rank 0 only) ----
        dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")
        out_dir = os.path.join(dora_dir, "temporal_logs") if dora_dir else "temporal_logs"
        os.makedirs(out_dir, exist_ok=True)
        record = {'epoch': epoch, 'windows': all_beat_stats}
        with open(os.path.join(out_dir, "windowed_audio_metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
 
        return metrics
 
    def evaluate_audio_generation(self) -> dict:
        """Evaluate audio generation with off-the-shelf metrics."""
        evaluate_stage_name = f'{self.current_stage}_generation'
        # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
        fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
        kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
        text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
        chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
        should_run_eval = False
        eval_chroma_wavs: tp.Optional[torch.Tensor] = None
        if self.cfg.evaluate.metrics.fad:
            fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.kld:
            kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.text_consistency:
            text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.chroma_cosine:
            chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
            # if we have predefind wavs for chroma we should purge them for computing the cosine metric
            has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
                                          self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
            if has_predefined_eval_chromas:
                warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
                                       'Resetting eval chromas to None for evaluation.')
                eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
                self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
            should_run_eval = True
 
        def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
            audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
            compressed_audio = self.compression_model.decode(audio_tokens, scale)
            return compressed_audio[..., :audio.shape[-1]]
 
        metrics: dict = {}
        if should_run_eval:
            loader = self.dataloaders['evaluate']
            updates = len(loader)
            lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
            average = flashy.averager()
            dataset = get_dataset_from_loader(loader)
            assert isinstance(dataset, AudioDataset)
            self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")
 
            # accumulators for windowed FAD + beat tracking
            do_windowed = getattr(self.cfg.evaluate, "windowed_audio_metrics", False)
            window_sizes_sec = list(getattr(self.cfg.evaluate, "window_sizes_sec", [10, 20, 30]))
            all_gen_wavs: tp.List[torch.Tensor] = []
            all_ref_wavs: tp.List[torch.Tensor] = []
 
            for idx, batch in enumerate(lp):
                audio, meta = batch
                assert all([self.cfg.sample_rate == m.sample_rate for m in meta])
 
                target_duration = audio.shape[-1] / self.cfg.sample_rate
                if self.cfg.evaluate.fixed_generation_duration:
                    target_duration = self.cfg.evaluate.fixed_generation_duration
 
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration,
                    remove_text_conditioning=self.cfg.evaluate.get('remove_text_conditioning', False)
                )
                y_pred = gen_outputs['gen_audio'].detach()
                y_pred = y_pred[..., :audio.shape[-1]]
 
                normalize_kwargs = dict(self.cfg.generate.audio)
                normalize_kwargs.pop('format', None)
                y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
                y = audio.cpu()  # should already be on CPU but just in case
                sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
                sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
                audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]
 
                if fad is not None:
                    if self.cfg.metrics.fad.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    fad.update(y_pred, y, sizes, sample_rates, audio_stems)
                if kldiv is not None:
                    if self.cfg.metrics.kld.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    kldiv.update(y_pred, y, sizes, sample_rates)
                if text_consistency is not None:
                    texts = [m.description for m in meta]
                    if self.cfg.metrics.text_consistency.use_gt:
                        y_pred = y
                    text_consistency.update(y_pred, texts, sizes, sample_rates)
                if chroma_cosine is not None:
                    if self.cfg.metrics.chroma_cosine.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    chroma_cosine.update(y_pred, y, sizes, sample_rates)
                    # restore chroma conditioner's eval chroma wavs
                    if eval_chroma_wavs is not None:
                        self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)
 
                # accumulate for windowed metrics (collect individual samples, not batches)
                if do_windowed:
                    for i in range(y_pred.shape[0]):
                        all_gen_wavs.append(y_pred[i].cpu())
                        all_ref_wavs.append(y[i].cpu())
 
            flashy.distrib.barrier()
 
            # ---- windowed FAD + beat tracking ----
            # NOTE: every rank MUST call _compute_windowed_fad_beat regardless of
            # whether its local shard is empty — the function performs an
            # all_gather_object internally, then dispatches the actual work to
            # rank 0 and broadcasts the result. Gating on local length here
            # would desync ranks.
            if do_windowed:
                windowed_metrics = self._compute_windowed_fad_beat(
                    gen_wavs=all_gen_wavs,
                    ref_wavs=all_ref_wavs,
                    sample_rate=self.cfg.sample_rate,
                    window_sizes_sec=window_sizes_sec,
                    epoch=self.epoch,
                )
                if windowed_metrics:
                    metrics.update(windowed_metrics)
 
            if fad is not None:
                metrics['fad'] = fad.compute()
            if kldiv is not None:
                kld_metrics = kldiv.compute()
                metrics.update(kld_metrics)
            if text_consistency is not None:
                metrics['text_consistency'] = text_consistency.compute()
            if chroma_cosine is not None:
                metrics['chroma_cosine'] = chroma_cosine.compute()
                
            metrics = average(metrics)
            metrics = flashy.distrib.average_metrics(metrics, len(loader))
 
        return metrics
    
    def evaluate(self) -> dict:
        """Evaluate stage."""
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid('evaluate'))
            gen_metrics = self.evaluate_audio_generation()
            return {**metrics, **gen_metrics}