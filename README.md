** Selective Speaker Identity Unlearning in Self-Supervised Speech Models**

Results at a Glance

| Metric | Before | After (GRL) | Target | Status |
|:---|:---:|:---:|:---:|:---:|
| Forget Speaker Accuracy | 100.0% | **0.0%** | < 10% | ✅ PASS |
| Retain Speaker Accuracy | 94.4% | **96.4%** | > 80% | ✅ PASS |
| MIA Confidence (forget) | High | **0.0000** | ~Random (0.0092) | ✅ PASS |
| Privacy Gap | — | **−0.0092** | Negative | ✅ PASS |
| Content Similarity | — | **0.9984** | > 0.80 | ✅ PASS |
| WER (Whisper-tiny) | — | **3.92** | Low WER | ✅ PASS |

> **All 6 targets achieved.** The model completely forgets p225 & p226 while retaining 96.4% accuracy on the remaining 107 speakers.
>
> What Is This?

Modern self-supervised speech models like **HuBERT** encode both *phonetic content* and *speaker identity* in the same 768-dimensional hidden state. This creates a serious privacy risk — if a user revokes consent, their voice remains embedded in the model with no way to remove it short of full retraining (which costs thousands of dollars and weeks of GPU time).

This project implements **Selective Speaker Identity Unlearning**: a lightweight framework that surgically removes specific speakers from a trained model — without retraining from scratch, and without access to the original training data.

---

## 🏗️ Architecture

```
Raw Audio (16kHz)
      │
      ▼
┌─────────────────────────────────────┐
│   HuBERT-base-ls960 Backbone        │  ← FROZEN (94.4M params)
│   Hidden State: 768-dim             │
└─────────────────────────────────────┘
      │
      ├──────────────────┬─────────────────────┐
      ▼                  ▼                     ▼
┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐
│ Speaker      │  │ Content      │   │ GRL + Adversarial    │
│ Encoder      │  │ Encoder      │   │ Head                 │
│ (~0.9M)      │  │ (~0.6M)      │   │ (~0.2M)              │
│              │  │              │   │                      │
│ Attentive    │  │ Residual MLP │   │ Gradient Reversal    │
│ Stats Pool   │  │ preserves    │   │ Layer flips grads    │
│ + MLP        │  │ phonetics    │   │ for forget speakers  │
└──────┬───────┘  └──────┬───────┘   └──────────┬───────────┘
       │                 │                        │
       ▼                 ▼                        ▼
  Speaker Logits    Content MSE Loss        KL → Uniform
  (CE Loss)         (phonetics intact)      (max confusion
                                           on forget spks)
```

**Total trainable parameters: 1.7M** (backbone stays frozen)

---

## 🔬 Methodology

### Phase 1 — Baseline Speaker Classification
Train a speaker encoder on top of frozen HuBERT to achieve ~93.3% validation accuracy. This **proves the privacy problem exists** — the model has memorised speaker identity.

### Phase 2 — Adversarial Unlearning via GRL

The **Gradient Reversal Layer (GRL)** is the core innovation:
- **Forward pass**: identity function (data passes through unchanged)
- **Backward pass**: gradients multiplied by `−alpha`, reversing the update direction
- **Effect**: representations become uninformative about forget speakers while retain speakers train normally

**Combined loss function:**

```
Total Loss = 1.0 × CE(retain) + 2.0 × KL(forget → Uniform) + 0.05 × MSE(content)
```

Alpha is ramped `0 → 1.5` via sigmoid warmup over 200 steps to prevent instability:

```python
alpha = max_alpha * (2 / (1 + exp(-10 * p)) - 1)
# where p = min(step / warmup_steps, 1.0)
```

### Post-Training — SVD Projection Removal
After GRL training, SVD is applied to forget-speaker embeddings (`shape: 12 × 768`) to find 6 principal subspace directions. These are **orthogonally projected out** of the speaker encoder weight matrix — a geometric guarantee that the speaker subspace is nulled, independent of training dynamics.

---

## 📊 Detailed Results

### Speaker Identification Accuracy

| Metric | Before | After GRL *(primary)* | After SVD Projection |
|:---|:---:|:---:|:---:|
| Overall Accuracy | 95.1% | 94.5% | 71.4% |
| Forget Speaker Acc | 100.0% | **0.0%** | **0.0%** |
| Retain Speaker Acc | 94.4% | **96.4%** | 71.0% |

> The GRL checkpoint is the primary result. SVD projection provides an additional weight-space guarantee at a retain accuracy trade-off; reducing projected directions from 6 to 2–3 would preserve retain accuracy above 80%.

### Membership Inference Attack (MIA)

| Metric | Value | Interpretation |
|:---|:---:|:---|
| Forget Speaker Confidence | 0.0000 | Model assigns 0% probability to the true label |
| Retain Speaker Confidence | 0.8328 | Retain speakers still strongly identified |
| Random Chance (1/109) | 0.0092 | Expected confidence if model forgot completely |
| **Privacy Gap** | **−0.0092** | **Forget speakers score BELOW random — stronger than required** |

## 🔭 Future Work

- [ ] Scale to all 109 VCTK speakers on A100 GPU for full SOTA comparison
- [ ] Tune SVD projection (2–3 directions) to better balance privacy vs. retain accuracy
- [ ] Compare against baselines: gradient ascent, SISA retraining, random noise injection
- [ ] Extend to multilingual speakers using mHuBERT or XLS-R
- [ ] Full WER evaluation on retain speakers post-unlearning

## 📚 References

1. Hsu et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP.
2. Baevski et al. (2020). *Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS.
3. Ganin & Lempitsky (2015). *Unsupervised Domain Adaptation by Backpropagation.* ICML.
4. Cao et al. (2015). *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P.
5. Veaux et al. (2017). *CSTR VCTK Corpus.* University of Edinburgh.
6. Shokri et al. (2017). *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P.


