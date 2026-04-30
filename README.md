# Mitigating Structural Collapse in Multi-Codebook Autoregressive Music Generation via Structure-Aware Scheduled Sampling

This repository contains the official PyTorch implementation for the ISMIR 2026 submission: **"Mitigating Structural Collapse in Multi-Codebook Autoregressive Music Generation via Structure-Aware Scheduled Sampling"**.

**Note:** This project is a customized fork of Meta's [AudioCraft](https://github.com/facebookresearch/audiocraft), specifically focused on the **MusicGen** model.

## Overview

Hierarchical autoregressive music models, such as MusicGen, frequently suffer from "structural collapse" during long-horizon generation. This degradation is caused by exposure bias compounding across the two-dimensional residual vector quantization (RVQ) grid over time and codebook depth. 

Standard scheduled sampling is ill-suited for RVQ-based models because replacing tokens independently across parallel codebooks creates unrealistic "chimeric" frames that violate inter-codebook dependencies. 

To solve this, we introduce **Structure-Aware Scheduled Sampling (SASS)**. SASS is a training intervention that dynamically replaces a subset of ground-truth frames with the model's own predictions, effectively bridging the train-inference gap while respecting the structural properties of the RVQ representation.

## Core Mechanisms

SASS addresses the fragility of the multi-codebook grid through four interlocking mechanisms:

*   **Temperature-Scaled Top-K Sampling:** Ensures the model explores plausible acoustic variations rather than injecting destructive, low-probability noise during training.
*   **Codebook-Weighted Confidence Gating:** Conditions frame replacement on the predictive reliability of fine-grained codebooks, ensuring that fragile acoustic details are not corrupted early in training.
*   **Dual-Axis Timing Gating:** Utilizes a warmup-then-decay schedule across training steps and temporal weighting across sequence time to target the late-stage regions where error accumulation is most severe.
*   **Joint Frame-Level Replacement:** Replaces all codebooks jointly per frame to preserve inter-codebook coherence and eliminate impossible chimeric mixtures.

## Key Results

Our framework is implemented over MusicGen and was extensively evaluated via controlled ablations and external zero-shot comparisons. 

*   **Improved Acoustic Fidelity:** SASS reduces Kullback-Leibler divergence by 5.6% to 10.0% across Forward, Reverse, and Symmetric KLD variants.
*   **Enhanced Rhythmic Stability:** Beat regularity is improved by up to 14.2% in the 10-20 second generation window.
*   **Flattened Temporal Degradation:** SASS significantly flattens the late-stage degradation slope (Delta FAD), narrowing the stability gap with non-autoregressive paradigms.
*   **Maintained Efficiency:** These structural improvements are achieved without altering the autoregressive factorization or tokenizer, preserving the high inference efficiency (Real-Time Factor) of the standard paradigm.
*   **Human Preference:** In A/B pairwise listening studies, raters strongly preferred SASS over standard Teacher Forcing and the 1.5B InspireMusic model for both musical coherence and rhythmic stability in 20-30s segments.

## Installation & Usage

AudioCraft requires Python 3.9, PyTorch 2.1.0. To install AudioCraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:

```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Data Preparation

AudioCraft expects the dataset to be organized under an `egs/` directory with separate folders for each split:

```
egs/jamendo/
├── train/
├── valid/
└── generate/
```

Each split folder contains paired `.wav` and `.json` files sharing the same filename. The `.wav` file is the audio clip, and the `.json` file holds its metadata. A typical `.json` file looks like:

```json
{
  "key": "",
  "artist": "Alexandria",
  "sample_rate": 44100,
  "file_extension": "wav",
  "description": "This instrumental track is categorized under the genres hard rock and rock, carrying an aura of darkness in its mood...",
  "keywords": "",
  "duration": 30.0,
  "bpm": "",
  "genre": "instrumental",
  "title": "Example Title",
  "name": "1000623_60",
  "instrument": "Mix",
  "moods": []
}
```

The `description` field is used as the text conditioning input during training. Make sure every `.wav` file has a corresponding `.json` file with at least the `description`, `sample_rate`, `file_extension`, `duration`, and `name` fields populated.

The `evaluate` and `generate` splits can point to the same folder if you want to use the same subset for both.

## Training with SASS

To fine-tune MusicGen-Small on the Jamendo dataset, run the following command. SASS is controlled by a dedicated group of flags — set `sass.enabled=true` to activate Structure-Aware Scheduled Sampling, or set it to `false` to fall back to standard Teacher Forcing.

### SASS-Specific Parameters

| Parameter | Description | Default |
|---|---|---|
| `sass.enabled` | Enable/disable SASS training | `false` |
| `sass.temp` | Temperature for top-k sampling during frame replacement | `1.0` |
| `sass.top_k` | Top-k filtering width for replacement token sampling | `250` |
| `sass.decay_ratio` | Fraction of training over which the replacement probability decays | `0.85` |

### Example Command

```bash
TS=$(date +%Y%m%d_%H%M%S)

export AUDIOCRAFT_REFERENCE_DIR=/path/to/Jamendo/reference
export AUDIOCRAFT_DORA_DIR=/path/to/dora_outputs/$TS
export PYTHONPATH=/path/to/audiocraft/google-research:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1 dora run -d \
  +xp.name=$TS \
  solver=musicgen/musicgen_base_32khz \
  model/lm/model_scale=small \
  continue_from=//pretrained/facebook/musicgen-small \
  dset=audio/jamendo \
  conditioner=text2music \
  optim.updates_per_epoch=1000 \
  optim.epochs=30 \
  optim.optimizer=adamw \
  optim.lr=1e-5 \
  dataset.batch_size=8 \
  dataset.segment_duration=30 \
  dataset.train.segment_duration=30 \
  dataset.valid.segment_duration=30 \
  dataset.evaluate.segment_duration=30 \
  dataset.generate.segment_duration=30 \
  dataset.train.num_samples=9000 \
  dataset.valid.num_samples=500 \
  dataset.evaluate.num_samples=100 \
  dataset.generate.num_samples=50 \
  evaluate.every=5 \
  generate.every=10 \
  evaluate.metrics.base=true \
  evaluate.metrics.fad=false \
  evaluate.metrics.kld=false \
  evaluate.metrics.text_consistency=false \
  evaluate.prefix_rollout_ce=false \
  evaluate.prefix_ratio=0.3 \
  evaluate.temporal_analysis=true \
  'evaluate.window_sizes_sec=[10,20,30]' \
  evaluate.windowed_audio_metrics=true \
  metrics.kld.passt.pretrained_length=30 \
  metrics.fad.tf.bin=/path/to/audiocraft/google-research \
  sass.enabled=true \
  evaluate.kld_tag=sass \
  evaluate.codebook_kld=true \
  sass.decay_ratio=0.85 \
  dataset.train.drop_desc_p=0.5 \
  classifier_free_guidance.training_dropout=0.3 \
  conditioners.description.t5.word_dropout=0.3 \
  optim.adam.weight_decay=0.1 \
  sass.temp=1.0 \
  sass.top_k=250 \
  seed=456
```

To run the baseline without SASS, simply set `sass.enabled=false` (or omit the SASS flags entirely).

## License

* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).

## Citation

If you use SASS in your research, please cite our paper:

```
@inproceedings{tsai2026sass,
    title={Mitigating Structural Collapse in Multi-Codebook Autoregressive Music Generation via Structure-Aware Scheduled Sampling},
    author={TODO},
    booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
    year={2026},
}
```

For the general framework of AudioCraft, please cite the following.

```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
