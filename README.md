# Advanced Data Science and Machine Learning for Health Research

Teaching materials for a 3-hour practical session in an **Advanced Data Science and Machine Learning for Health Research** summer course.

This repository contains two independent Google Colab notebooks designed for clinical researchers, epidemiologists, biomedical scientists, and data scientists with prior experience in Python, basic machine learning, train/validation/test splitting, cross-validation, overfitting, regularization, and basic neural networks.

The practical sits after lectures on:

- neural networks and convolutional neural networks
- autoencoders
- diffusion models

and before a later lecture on transformers.

## Practical Overview

The session is split into two notebooks:

1. `01_CNN_Explainability_Health.ipynb`
2. `02_Autoencoder_Latent_Space_Health.ipynb`

Each notebook is designed for roughly **75-90 minutes** and combines:

- concise theory
- executable code
- visual outputs
- interpretation questions
- short coding exercises
- health-research discussion

The emphasis is on **model exploration, critical interpretation, and external validity**, not on large-scale training.

## Notebooks

### Practical 1

**What Is a Convolutional Neural Network Looking At?**

Topics include:

- hierarchical CNN representations
- feature maps
- pretrained convolutional networks
- Grad-CAM
- perturbation-based analysis
- shortcut learning in medical imaging
- confounding, domain shift, and external validity

Open in Colab:

- [Practical 1 in Colab](https://colab.research.google.com/github/roshchupkin/Advanced_ML_2026/blob/main/01_CNN_Explainability_Health.ipynb)

### Practical 2

**What Does an Autoencoder Learn About Patients?**

Topics include:

- encoder-decoder architectures
- bottlenecks and reconstruction loss
- latent representations
- PCA in original and latent spaces
- simulated site effects and batch/domain shift
- latent interpolation
- downstream prediction from learned features
- conceptual links to latent diffusion

Open in Colab:

- [Practical 2 in Colab](https://colab.research.google.com/github/roshchupkin/Advanced_ML_2026/blob/main/02_Autoencoder_Latent_Space_Health.ipynb)

## Computational Design

These materials were written specifically for **Google Colab**, assuming the worst case of a **CPU-only runtime**.

Core design constraints:

- no GPU is required
- no Google Drive mount is required
- no Kaggle access is required
- no institutional credentials are required
- no manual data upload is required
- no private datasets are used

Approximate practical-wide requirements:

- downloads: about **101 MB** for the core path
- peak RAM: about **2-2.5 GB**
- heavy compute: about **6-10 CPU minutes** total

All datasets are downloaded automatically from public sources. Any optional package installation is clearly separated from the core teaching path.

## Datasets

The notebooks use lightweight public biomedical datasets from **MedMNIST / MedMNIST+**:

- **PneumoniaMNIST (64x64)** for the CNN and explainability practical
- **BloodMNIST (28x28)** for the autoencoder and latent-space practical

These datasets are intended for **research and education**, not clinical use.

## Repository Files

- `01_CNN_Explainability_Health.ipynb` — student practical on CNNs and explainability
- `02_Autoencoder_Latent_Space_Health.ipynb` — student practical on autoencoders and latent representations
- `README_Instructor.md` — instructor-facing guide with timings, compute profile, citations, pre-course checks, and complete exercise solutions

## For Instructors

If you are teaching from this repository, start with:

- `README_Instructor.md`

It contains:

- timing tables
- expected runtime and memory profile
- package and license notes
- data and model citations
- a pre-course Colab smoke-test checklist
- known failure modes
- full solutions for all exercises

## Citation

If you reuse the teaching materials, please cite the underlying datasets and models as described in the notebooks and in `README_Instructor.md`.

Key dataset references include:

- Yang et al., *Scientific Data* 2023, MedMNIST v2
- Yang et al., *ISBI* 2021, MedMNIST Classification Decathlon
- Kermany et al., *Cell* 2018, source study for PneumoniaMNIST
- Acevedo et al., *Data in Brief* 2020, source study for BloodMNIST

## Notes

- The notebooks intentionally ship **without stored outputs**.
- Simulated shortcut-learning and simulated site effects are explicitly labelled inside the notebooks.
- The materials are designed to encourage students to ask not only whether a model predicts well, but also **what information it is actually using**.
