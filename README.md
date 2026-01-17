Privacy-Preserving Financial Fraud Detection with PyTorch GANs
📌 Project Overview
Financial institutions face a major challenge: sharing data for fraud research without violating PII (Personally Identifiable Information) regulations. This project implements a Generative Adversarial Network (GAN) to synthesize anonymized financial transactions that retain the statistical "signals" of fraud while protecting user privacy.

🚀 Key Performance Metrics
Model Utility (TSTR): Achieved an ROC-AUC of 0.78 on real-world test data using a model trained exclusively on synthetic samples.

Fraud Detection Recall: 99%, demonstrating that the GAN successfully captured even the most subtle fraud patterns.

Privacy Guarantee: Zero data leakage confirmed via Distance-to-Closest-Record (DCR) analysis.

🛠️ Technical Architecture
Pre-processing: Implemented log-transformations for skewed transaction amounts and cyclical encoding for time-based features.

Custom GAN: Built a deep-layered Generator and Discriminator using PyTorch with LeakyReLU and BatchNorm to ensure training stability.

Validation: Employed a "Train-Synthetic, Test-Real" (TSTR) methodology to validate the downstream value of the generated data.

📈 Visual Validation
Distribution Matching: The synthetic data (Orange) successfully mirrors the multi-modal distributions of the original transaction logs (Blue).

Privacy Buffer: Statistical analysis confirms that no synthetic record is an exact replica of a real-world entry.
