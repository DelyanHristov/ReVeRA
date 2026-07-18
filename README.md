# ReVeRA — Parameter-Efficient Pretraining through Low-Rank and High-Rank Adaptation

> **Parameter-Efficient Pretraining through Low-Rank and High-Rank Adaptation Techniques**
> Delyan Hristov · Mentor: Radostin Cholakov

ReVeRA is a parameter-efficient **pre-training** method that combines the ideas of
[VeRA](https://arxiv.org/abs/2310.11454) and [ReLoRA](https://arxiv.org/abs/2307.05695).
Like VeRA, it keeps a single pair of low-rank matrices `A` and `B` **frozen and shared across all
layers**, and only trains two small scaling vectors `Λ_b` and `Λ_d` (treated as diagonal matrices).
Like ReLoRA, these adapters are **merged into the frozen weights and re-initialized** every few
thousand steps, which turns a sequence of low-rank updates into a high-rank update:

```
ΔW = α·Λ¹_b A Λ¹_d B + α·Λ²_b A Λ²_d B + ··· + α·Λᴺ_b A Λᴺ_d B
```

where `α = ReVeRA_alpha / r`. Because only the scaling vectors are trainable, ReVeRA uses **far
fewer trainable parameters than ReLoRA** (e.g. ~13.8K vs. ~1.57M at `r = 64` on BERT-medium),
which makes it attractive when memory is the binding constraint and you are willing to trade some
accuracy for it.

This repository is a fork of the official ReLoRA codebase
([Guitaricet/peft_pretraining](https://github.com/Guitaricet/peft_pretraining)), adapted to train
**BERT-medium** with masked language modeling on the **C4** corpus. The training loop, the
merge-and-reinitialize logic, and the cyclical cosine learning-rate schedule are inherited from
ReLoRA, so most of the ReLoRA flags below still apply.

> **Note on naming:** the code keeps ReLoRA's original argument names. When training ReVeRA,
> `--lora_r` is the rank `r`, `--lora_alpha` is `ReVeRA_alpha`, and `--relora` / `--cycle_length`
> control how often the adapters are merged and reset.

## Setup

Requires Python 3.10+ and a recent PyTorch.

```bash
pip install -e .
```

All requirements are listed in `requirements.txt`.

## How it works

- The adapters are injected into the attention `query`, `key`, and `value` layers.
- `A` and `B` are Kaiming-initialized and frozen; the scaling vectors are initialized so the
  adapter output is zero at the start of each cycle (`Λ_b = 0`, `Λ_d = 10⁻¹`), so the model behaves
  exactly like the base model right after every reset.
- Every `--relora` update steps the adapters are merged into the frozen weights, a percentage of
  the optimizer state is reset, the learning rate is dropped to ~0 and warmed back up over
  `--restart_warmup_steps` (see the cyclical schedule below). This is what produces the high-rank
  update.

## 1. Pre-process the data

C4 is tokenized ahead of time with the BERT-medium tokenizer:

```bash
python pretokenize.py \
    --save_dir preprocessed_data \
    --tokenizer prajjwal1/bert-medium \
    --dataset c4 \
    --dataset_config en \
    --text_field text \
    --sequence_length 512
```

The script logs where the pre-processed data is saved (something like
`preprocessed_data/c4_prajjwal1-bert-medium_512`). Point `--dataset_path` at that directory when
training.

## 2. Pre-train with ReVeRA

Passing `--model_name_or_path` (any value, e.g. `bert_c4`) builds the hardcoded BERT-medium
configuration (hidden size 512, 8 layers, 8 heads) and trains it with masked language modeling.
Enable the adapters with `--use_peft True`.

```bash
export DATA_PATH=<path to preprocessed data>

torchrun --nproc_per_node=1 torchrun_main.py \
    --model_name_or_path bert_c4 \
    --dataset_path $DATA_PATH \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 64 \
    --relora 15625 \
    --cycle_length 15625 \
    --scheduler cosine_restarts \
    --warmup_steps 9000 \
    --restart_warmup_steps 500 \
    --reset_optimizer_on_relora False \
    --optimizer_magnitude_pruning 0.9 \
    --lr 1e-3 \
    --weight_decay 0.02 \
    --batch_size 64 \
    --max_length 512 \
    --num_training_steps 62500 \
    --eval_every 3500 \
    --save_every 600000 \
    --seed 42 \
    --workers 4 \
    --tags revera_bert_medium
```

To train a plain **VeRA** baseline (no merging/resetting), set `--use_peft True` with
`--scheduler linear` and `--relora` equal to the total number of steps so no reset happens. To
train the **ReLoRA / LoRA** baselines, use the original ReLoRA implementation with trainable `A`/`B`.

## Hyperparameters

The optimal configuration reported in the paper (BERT-medium, `r = 64`, batch size 64,
bfloat16):

| Hyperparameter        | Value  | CLI flag                  |
| --------------------- | ------ | ------------------------- |
| learning rate         | 3e-3   | `--lr`                    |
| weight decay          | 2e-2   | `--weight_decay`          |
| warmup steps          | 9000   | `--warmup_steps`          |
| cycle length          | 15625  | `--cycle_length` / `--relora` |
| restart warmup steps  | 500    | `--restart_warmup_steps`  |
| ReVeRA alpha          | 256    | `--lora_alpha`            |
| ReVeRA dropout        | 0.1    | (adapter default)         |

`cycle length` is the number of steps between each merge-and-reinitialize of the adapters;
`restart warmup steps` is the warmup applied right after every merge.

## Optimizer resets and the LR schedule

These are inherited from ReLoRA and matter for getting a genuine high-rank update:

1. **Optimizer state** — resetting it prevents Adam's stored moments from cancelling the effect of
   the reset. Options:
   ```
   --reset_optimizer_on_relora   (default True)
   --optimizer_random_pruning    (float, default 0.0)
   --optimizer_magnitude_pruning (float, default 0.0)
   ```
   The runs in this project use `--reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.9`.
2. **Learning rate** — ReVeRA only supports `--scheduler cosine_restarts`, which drops the LR to ~0
   at each reset and warms it back up over `--restart_warmup_steps` (see figure in the paper).
3. **Reset frequency** — set by `--relora` (in update steps).

## Results

C4 pre-training, `r = 64`, bfloat16, evaluated by perplexity (lower is better):

| Method  | Trainable params | Perplexity |
| ------- | ---------------- | ---------- |
| LoRA    | 1572.8K          | 30.40      |
| ReLoRA  | 1572.8K          | **18.41**  |
| VeRA    | 13.8K            | 78.59      |
| ReVeRA  | 13.8K            | 45.93      |

ReVeRA clearly beats plain VeRA (confirming that merging yields a high-rank update) and uses two
orders of magnitude fewer trainable parameters than ReLoRA, at the cost of some perplexity.

Trainable-parameter counts scale favorably with model size — see Table 1 in the paper (e.g. on
GPT-3, ReVeRA stays ~3.5M trainable across `r ∈ {16, 64, 256}`, while ReLoRA grows from 67.8M to
1811.9M).

## Fine-tuning

As with ReLoRA, the merge-and-reinitialize process was found **not** to help for fine-tuning. On the
GLUE/COLA task with T5-base (`run_glue.py`), ReVeRA slightly underperforms plain VeRA across
`r ∈ {256, 512, 1024}` (see Appendix B of the paper). ReVeRA is intended for **pre-training**.

## Acknowledgments

This work builds directly on [ReLoRA](https://arxiv.org/abs/2307.05695) and
[VeRA](https://arxiv.org/abs/2310.11454). Thanks to Radostin Cholakov for proposing the topic and
for guidance, and to Nikola Gyulev, Delyan Boychev, and Emiliana Dimitrova for feedback. The
America for Bulgaria and Beautiful Science foundations partially funded the compute, and the Google
ML Developer Programs team provided Google Cloud and Colab credits.

## Citation

If you use this work, please cite the ReVeRA report along with the ReLoRA and VeRA papers it builds
on:

```
@misc{hristov_revera,
    title={Parameter-Efficient Pretraining through Low-Rank and High-Rank Adaptation Techniques},
    author={Delyan Hristov},
    note={Mentor: Radostin Cholakov}
}

@misc{lialin2023stack,
    title={Stack More Layers Differently: High-Rank Training Through Low-Rank Updates},
    author={Vladislav Lialin and Namrata Shivagunde and Sherin Muckatira and Anna Rumshisky},
    year={2023},
    eprint={2307.05695},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@misc{kopiczko2024vera,
    title={VeRA: Vector-based Random Matrix Adaptation},
    author={Dawid J. Kopiczko and Tijmen Blankevoort and Yuki M. Asano},
    year={2024},
    eprint={2310.11454},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
