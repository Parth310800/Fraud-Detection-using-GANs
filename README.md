# Privacy–Utility Trade-off in GAN-Generated Financial Transaction Data

A GAN trained on PaySim transaction data produces synthetic records that leak no real
customer data — and lose almost all of their fraud-detection value once you measure
them the right way.

The headline finding is the gap between two ways of scoring the same synthetic dataset:

| Measurement                         | Synthetic-trained | Real-trained (ceiling) | Retention |
|-------------------------------------|-------------------|------------------------|-----------|
| ROC-AUC, balanced test set          | 0.866             | 0.999                  | **86.7%** |
| PR-AUC, true 0.13% prevalence       | 0.006             | 0.748                  | **0.8%**  |

Both rows describe the same model on the same held-out data. The first is the number
the standard TSTR protocol reports. The second is what the data is actually worth.

---

## Why the gap exists

Fraud is 0.13% of PaySim transactions. Under that imbalance, ROC-AUC is dominated by
the enormous, trivially-separable negative class: the false-positive rate barely moves
even when the model flags hundreds of thousands of legitimate transactions. PR-AUC
does not have that escape hatch, because precision is computed against the tiny
positive class.

In operational terms, on 1.27M held-out transactions:

- **Synthetic-trained:** catches 99.9% of fraud, but non-fraud recall is 46% — roughly
  683,000 false alarms to find ~1,642 frauds. Precision ≈ 0.24%.
- **Real-trained:** catches 99.2% of fraud with 98% non-fraud recall.

The synthetic-trained model isn't detecting fraud. It's flagging most of the dataset
and being credited for the frauds swept up along the way.

## The privacy result

Distance-to-closest-record, computed in scaled feature space against a real-to-real
baseline:

| | Min | Median |
|---|---|---|
| Synthetic → real | 0.0012 | **0.0909** |
| Real → real (reference) | 0.0000 | 0.0034 |

Synthetic records sit ~27× further from real records than real records sit from each
other, with zero exact replicas. No memorisation is evident.

Note that the real-to-real minimum is 0.0 — the source data contains duplicate rows.
DCR is one check against memorisation, not a formal privacy guarantee; it says nothing
about attribute inference or membership inference risk.

## Reading both results together

Strong privacy and weak utility are one finding seen twice. The generator learned a
distribution that does not hug the real data manifold — which is exactly what protects
privacy, and exactly what destroys the fine-grained structure a fraud classifier needs.

The practical takeaway: **evaluating synthetic tabular data with TSTR + ROC-AUC on a
balanced test set will substantially overstate its utility under class imbalance.**
Report PR-AUC at true prevalence, and always against a train-real baseline.

---

## Method

- **Data:** PaySim (6,362,620 transactions, 8,213 fraud). A stratified 20% test set is
  held out *before* any downsampling, so the evaluation set keeps true prevalence and
  the GAN never sees it.
- **Features:** log-transformed amount, four balance fields, and cyclical sine/cosine
  encoding of transaction hour.
- **GAN:** custom PyTorch generator (100-dim latent → 128 → 256 → 512 → 8, LeakyReLU,
  BatchNorm, Tanh) and discriminator, BCE loss, Adam, 100 epochs on 13,140 balanced
  rows from the dev split. Seeded for reproducibility.
- **Evaluation:** identical `RandomForestClassifier` trained once on synthetic data
  (TSTR) and once on real data (TRTR), scored on the same held-out test set at both
  balanced and true prevalence. The only variable is the origin of the training rows.

## Limitations

- A single GAN architecture and one seed; no comparison against CTGAN, TVAE, or a
  conditional GAN, any of which may close the utility gap.
- `isFraud` is generated continuous and thresholded, rather than modelled as a
  conditional label — a likely contributor to the poor utility.
- DCR is computed against the GAN's training rows only, so it does not test whether
  synthetic points sit closer to training data than to unseen real records.
- No formal privacy accounting (e.g. differential privacy) is applied or claimed.

## Revision note

An earlier version of this project reported ROC-AUC 0.78 as its headline result and
described 99% fraud recall as evidence of a strong model. That evaluation used a
50/50 test set, reported no train-real baseline, and omitted PR-AUC. Once a TRTR
baseline and true-prevalence PR-AUC were added, the utility claim did not survive.
The current results are the corrected ones.

## Reproducing

```bash
pip install -r requirements.txt
```

Download the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place
`PS_20174392719_1491204439457_log.csv` in the repository root (the CSV is ~470 MB and
is gitignored). Then run `fraud_detection_gan.ipynb` top to bottom. Results are written
to `results_gan.json`.
