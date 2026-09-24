# Handwritten Maths Expression Recognition

Deep learning pipeline that converts handwritten stroke data into mathematical
expressions, built for the UCD "Foundations of Deep Learning" final assignment
(Winter 2025). The project progresses through three parts, each implementing a
classic architecture family on the same stroke-based input format:

| Part | Task | Architecture |
|---|---|---|
| 1 | Glyph classification (18 classes) | CNN with BatchNorm |
| 2 | Infix expression recognition (seq2seq) | BiLSTM encoder, LSTM decoder, Bahdanau attention |
| 3 | Postfix expression recognition (seq2seq) | Transformer encoder-decoder |

**Headline results** (trained and evaluated on the full course datasets, RTX 3050 laptop GPU):

| Part | Dataset | Test metric | Result |
|---|---|---|---|
| 1 | glyph_80k.h5 (80k samples) | Accuracy / F1 | **83.12% / 0.83** |
| 2 | infix_88k.h5 (88k samples) | Sequence accuracy | **64.95%** (model 2) |
| 3 | postfix_208k.h5 (208k samples) | Token accuracy | **93.34%** |

> **Code availability.** The implementation notebooks are intentionally **not
> published** in this repository, in line with university academic integrity
> policy for coursework. This repo presents the methodology, trained
> checkpoints and results; the source notebooks are available on request.

Full methodology, architecture diagrams, challenges and error analysis are in
[`docs/DL_Assignment_Report.docx`](docs/DL_Assignment_Report.docx); the original
brief is in [`docs/ASSIGNMENT_README.pdf`](docs/ASSIGNMENT_README.pdf).

## Repository layout

```
final_project/
├── models/                          # trained checkpoints (best of each run)
│   ├── part1_glyph_cnn.pth                 # epoch 24, val loss 1.089, val acc 83.34%
│   ├── part2_infix_attention_model1.pth    # ~5 epochs, test seq acc 53.2%
│   ├── part2_infix_attention_model2.pth    # trained longer, test seq acc 64.95%
│   └── part3_postfix_transformer.pth       # val loss 0.226, test token acc 93.34%
├── results/                         # training curves, confusion matrices, metrics
├── datasets/                        # bundled 1k debug sets (see Datasets section)
├── docs/                            # assignment brief (PDF) + full report (docx)
├── requirements.txt
└── README.md

notebooks/                           # implementation (kept private, not published)
```

## Datasets

All three parts consume the same stroke-based H5 format: inputs are 2D pen
coordinates (`[seq_len, 128]` float32, padded with -5, with `<bos>`/`<eos>` stroke
markers), outputs are tokenised expression strings.

| File | Size | Purpose |
|---|---|---|
| `glyph_1k.h5`, `infix_1k.h5`, `postfix_1k.h5` | ~6 MB total | Debug sets, used for pipeline bring-up |
| `glyph_80k.h5` | ~30 MB | Part 1 full training set |
| `infix_88k.h5` | ~240 MB | Part 2 full training set |
| `postfix_208k.h5` | ~568 MB | Part 3 full training set, exceeds GitHub's 100 MB file limit |

The full datasets are course materials and are not redistributed here; only the
three 1k debug sets are included.

## Part 1: CNN for Glyph Classification

**Task.** Classify a single handwritten glyph (digits 0-9 and symbols
`+ - * / . ( ) =`) into 18 classes.

**Architecture** (2D strokes kept as a pseudo-image, no flattening):

```
Input [batch, strokes, 128]
  → Conv2d(1→16, 3x3, pad 1) → BatchNorm → ReLU → MaxPool(2,2)
  → Conv2d(16→32, 3x3, pad 1) → BatchNorm → ReLU → MaxPool(2,2)
  → Flatten → Linear(1024→128) → ReLU → Dropout(0.6) → Linear(128→18)
```

**Training.** Adam (lr 0.001, weight decay 1e-4), `ReduceLROnPlateau` on val loss,
batch size 128, 50 epochs with best-checkpoint selection on validation accuracy.

**Results.** Best epoch 24 (val loss 1.089, val acc 83.34%): test accuracy
**83.12%**, macro F1 **0.83**. The confusion matrix
([`results/part1_confusion_matrix.png`](results/part1_confusion_matrix.png)) shows
errors concentrated in visually confusable pairs (e.g. 1 vs 7, and digit vs
symbol mix-ups on rarely drawn symbols). Training took about 2 hours on the
RTX 3050.

Curves: [`results/part1_training_curves.png`](results/part1_training_curves.png)

## Part 2: LSTM Encoder-Decoder with Bahdanau Attention for Infix

**Task.** Map a stroke sequence to an infix expression (e.g. `4*(5.3+2)=`),
a sequence-to-sequence problem over a 23-token vocabulary (18 base + `<unk> <pad> <bos> <eos>`).

**Architecture:**

```
Encoder: Linear(1→128) → 1-layer BiLSTM (hidden 128, dropout 0.3) → 256-d outputs/states
Decoder: Embedding(23→64) → 1-layer LSTM (hidden 256) with additive
         (Bahdanau) attention over encoder outputs → Linear(→23)
Hidden/cell transforms bridge the bidirectional encoder states to decoder dims.
```

**Training.** Teacher forcing ratio 0.9, Adam (lr 0.001, weight decay 1e-4),
`ReduceLROnPlateau` (factor 0.5, patience 5), gradient clipping (max norm 1.0),
CrossEntropy with `ignore_index` for padding, batch size 64.

**Results.** Two models were trained:
- **Model 1** (~5 epochs): test seq accuracy **53.22%**, token accuracy 90.58%, test loss 0.417
- **Model 2** (same architecture, trained longer over multiple sessions): test seq accuracy **64.95%**,
  token accuracy 93.15%, test loss 0.385, the headline Part 2 number.

Typical failure mode: single-token slips on visually similar digits, e.g.
ground truth `2.4/2.5*3/1=` predicted as `2.4/7.5*3/1=`.

Curves: [`results/part2_training_history.png`](results/part2_training_history.png)

## Part 3: Transformer Encoder-Decoder for Postfix

**Task.** Map stroke sequences to postfix / Reverse Polish Notation (e.g. infix
`3+5*2=` becomes postfix `3,5,2,*,+,=`, where `,` marks end-of-number). Vocabulary
of 23 tokens. Largest dataset of the three (208k samples).

**Architecture** (PyTorch `nn.Transformer`):

```
Encoder: Linear(1→d_model 128) + sinusoidal positional encoding
         → 4 self-attention layers (4 heads, FFN 512, dropout 0.1)
Decoder: Embedding(23→128) + positional encoding → 4 masked self-attention
         layers with cross-attention to the encoder → Linear(→23)
Padding masks on both sides, causal mask on the decoder.
```

**Training.** Adam (lr 1e-4, weight decay 1e-5), StepLR (halve every 10 epochs),
batch size 128, CrossEntropy with `ignore_index=-5`, checkpoint on best val loss,
about 24 hours total on the RTX 3050. (The assignment report describes an
"Attention is All You Need" warm-up schedule; the final run used the simpler
StepLR schedule above.)

**Results.** Test token accuracy **93.34%**, exact-match sequence accuracy 4.2%.
The token vs sequence gap is expected: with ~15-token outputs, even uncorrelated
93% per-token accuracy implies about 0.93^15 ≈ 34% perfect sequences, and errors
correlate in practice. The model was still improving when training was stopped
for time reasons.

Curves: [`results/part3_training_curves.png`](results/part3_training_curves.png) ·
per-epoch metrics: [`results/training_metrics.json`](results/training_metrics.json)

## Checkpoints

`models/` contains the best checkpoint of each run:
- `part1_glyph_cnn.pth`: dict with `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_loss`, `val_acc`
- `part2_infix_attention_model1.pth`, `part2_infix_attention_model2.pth`,
  `part3_postfix_transformer.pth`: bare `state_dict`s

Loading them requires the model class definitions, which live in the private
notebooks; the checkpoints are included here as evidence of the reported runs.

## Tooling notes

- A custom `Vocabulary` class replaces `torchtext.vocab.Vocab` throughout, since
  torchtext is incompatible with modern PyTorch (the course startup notebooks used it).
- Training targeted a single consumer GPU (RTX 3050 laptop): Part 1 ~2 h,
  Part 2 ~7 h/epoch (several days total), Part 3 ~24 h.

## License

Course assignment materials (datasets, assignment brief) remain the property of
the course. This repository presents coursework results for portfolio purposes.