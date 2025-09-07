# Methodology

## Approach
Add a self-critical RL (SCST) phase on top of the supervised BART baseline to directly optimize a penalized BLEU reward:

`R = BLEU(ref, hyp) × (100 − BLEU(input, hyp)) / 52`

For each batch, decode a greedy baseline (no sampling) and a stochastic sample (top-p/top-k/temperature), compute per-example rewards, and use the advantage `A = R(sample) − R(greedy)` for a policy-gradient update. Mix a small MLE stabilizer with the RL loss to improve stability.

## Expectation
- Improve penalized BLEU over pure MLE by rewarding faithfulness to the reference while discouraging near-copies of the source.  
- Increase lexical/syntactic diversity without sacrificing fluency.  
- Stabilize training via an MLE stabilizer, advantage normalization, variance guard, gradient clipping, and consistent BLEU settings.

## Changes

### 1. SCST objective on penalized BLEU
- Sentence-level SacreBLEU with `effective_order=True` (default smoothing) used for the **per-sample** reward; advantage `A = R(sample) − R(greedy)`.  
- Decoding split: greedy baseline vs sampled rollout with these sampled rollout hyperparameters:
  - `do_sample=True`
  - `top_p=0.9`
  - `top_k=50`
  - `temperature=1.5`
  - `min_length=8`
  - `repetition_penalty=1.1`

### 2. Stability and optimization controls
- Loss mixing: `loss = 0.3 × PG + 0.7 × MLE` (λ_rl tunable).  
- Advantage normalization with floor: `scale = max(std(A), 0.1)`; skip PG when `std(A) < 1e−6`.  
- Gradient clipping at 1.0.  
- Careful train/eval toggling around `generate()` calls and during log-prob computation to ensure correct `requires_grad` semantics.

### 3. Metric and checkpoint discipline
- Reward/eval consistency: use the same BLEU flavor (SacreBLEU, `effective_order=True`) during reward computation to avoid metric drift; corpus-level BLEU is used for final reporting.  
- Checkpoint-on-improvement: evaluate dev penalized BLEU each RL epoch and overwrite `best_bart_model.pt` only when the dev penalized BLEU improves; reload the best checkpoint at the end.

---

## Results

- **Validation (dev, penalized BLEU):** `26.215 → 31.797` (+5.582 absolute, +21.3% relative).  
- **Relative to earlier MLE baseline (24.754):** `+7.043` absolute, `+28.5%` relative.  
- Observed trend: best results commonly occur early in RL (example best at epoch 2/6); later epochs can oscillate but typically remain above the pre-RL checkpoint.  
- Qualitative: fewer near-copies of the source; more varied phrasing and structure while preserving meaning.

**Per-epoch summary (dev metrics)**

| Phase/Epoch | BLEU(ref→hyp) | 100−BLEU(input→hyp) | *Penalized BLEU* |
|---|---:|---:|---:|
| Supervised (before RL) | 42.290 | 32.234 | *26.215* |
| RL Epoch 1 | 41.490 | 34.487 | *27.516* |
| *RL Epoch 2 (best)* | *38.506* | *42.939* | *31.797* |
| RL Epoch 3 | 38.680 | 41.769 | *31.069* |
| RL Epoch 4 | 39.272 | 40.896 | *30.886* |
| RL Epoch 5 | 37.558 | 43.463 | *31.392* |
| RL Epoch 6 | 39.520 | 39.660 | *30.142* |

---

## Discussion

### Why these results
- **Objective alignment:** Penalized BLEU directly matches the paraphrase evaluation goal and explicitly penalizes copying from the input—this addresses a major failure mode on high-overlap paraphrase datasets.  
- **Variance reduction:** The self-critical advantage (`sample − greedy`) cancels shared biases between the two trajectories, producing lower-variance, more useful policy gradients.  
- **Reward smoothing:** Sentence-level SacreBLEU smoothing avoids zero n-gram rewards on medium-length outputs, which stabilizes policy updates.  
- **Controlled exploration:** Top-p/top-k/temperature sampling lets the policy explore realistic paraphrase alternatives; SCST only reinforces sampled outputs that outperform greedy under the reward, converting exploration into consistent metric gains.

### Sampling rationale (short)
- `top_p = 0.9`: keeps 90% probability mass and surfaces viable alternatives while avoiding extreme tails.  
- `top_k = 50`: caps candidate tokens to avoid sampling extremely low-probability tokens when the distribution is flat.  
- `temperature = 1.5`: flattens the distribution enough for synonyms/alternate constructions to appear without breaking fluency.  
- `min_length = 8`: prevents very short outputs and allows room for structural variation.  
- `repetition_penalty = 1.1`: discourages verbatim copying and loops, nudging towards lexical variety.  
- `do_sample = True` + `early_stopping = False`: ensures genuine exploration beyond greedy and allows sequences to realize alternative structures before EOS.

### What it achieves
- Higher penalized BLEU vs. pure MLE by balancing faithfulness and novelty.  
- Increased lexical and syntactic diversity while preserving fluency (thanks to MLE mixing and repetition/length controls).  
- More stable RL training via advantage normalization, variance guards, and gradient clipping.  
- Reproducible best-model selection using dev penalized BLEU checkpointing.

### What it could not fully achieve (and why)
- **Monotonic improvement across epochs:** sentence-level rewards are noisy; improvements often peak early and can oscillate thereafter.  
- **Perfect semantic adequacy control:** penalized BLEU is only a proxy for meaning—overly aggressive sampling can still harm semantics.  
- **Hyperparameter robustness:** outcomes depend on sampling hyperparameters, `λ_rl`, RL learning rate; mis-tuning can negate gains.  
- **Full train/eval parity:** training uses sentence-level smoothed BLEU for stability while reporting uses corpus BLEU for comparability; this intentional mismatch cannot be fully eliminated but is managed.

---

## Reproducibility notes
- Use SacreBLEU with `effective_order=True` both in reward computation and when possible in final evaluation to reduce metric mismatch.  
- Keep advantage normalization and the variance guard (`skip PG if std(A) < 1e−6`) to avoid degenerate PG updates.  
- Mix a stable MLE term (e.g., λ_rl = 0.3) to preserve fluency.  
- Save the best model only when dev penalized BLEU improves; reload at the end for downstream inference.

---

## Padding Masking in Targets

### Approach
During supervised (MLE) training and in the MLE stabilizer used during RL, mask all padding tokens in the target labels so they do not contribute to the loss. Implementation: set `labels[labels == tokenizer.pad_token_id] = -100`, which PyTorch’s `CrossEntropyLoss` ignores by default.

### Expectation
- Prevents the model from learning to predict `<pad>` tokens, reduces noise in the loss, and improves BLEU scores.  
- Reduces length bias (the model over-predicting `<pad>` or ending too early).  
- Produces cleaner gradients for faster, more stable convergence and improved penalized BLEU.  
- In RL, ensures the MLE stabilizer reinforces content tokens rather than padding.

### Changes
- **Supervised dataloader (`transform_data`)**: after tokenizing targets, clone to `labels` and set pad positions to `-100`.  
- **RL epoch (`rl_finetune_epoch`)**: re-tokenize refs per batch for the stabilizer and again set pad positions to `-100`.  
- Verified `tokenizer.pad_token_id` and generation args (`eos_token_id`, `no_repeat_ngram_size`) are correctly set.

### Results
- Training: lower and smoother training loss curves; reduced variance across steps.  
- Qualitative: fewer truncated outputs; more stable sequence lengths; fewer padding artifacts in generations.  
- Validation: penalized BLEU improved from **20** to **26** on the dev set with identical decoding settings.

### Discussion
Masking padding is standard for seq2seq cross-entropy; without it, the model “learns” to predict padding and shortens outputs. This change helps decouple sequence length control from loss shaping and complements decoding constraints (e.g., `no_repeat_ngram_size`, min/max length). In the SCST setup, it keeps the MLE component aligned with semantic content, avoiding distortions from padding-heavy batches.

**Caveats:** ensure the correct pad ID (especially if swapping tokenizers); do not reintroduce padding loss via a custom collator; if using label smoothing, apply it after masking so pads remain ignored.
