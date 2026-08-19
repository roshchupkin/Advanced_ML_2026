# Instructor Guide

## [Advanced Data Science and Machine Learning for Health Research](https://erasmussummerprogramme.nl/summer-programme-courses/advanced-data-science-and-machine-learning-for-health-research) - 3-hour practical

| File | Content | Duration |
|---|---|---|
| `01_CNN_Explainability_Health.ipynb` | Practical 1: CNNs and explainability in medical imaging (PneumoniaMNIST) | 80-90 min |
| `02_Autoencoder_Latent_Space_Health.ipynb` | Practical 2: *What Is Hidden in a Learned Representation?* (BloodMNIST) | 80-85 min |
| `weights/` | Instructor fallback checkpoints (`TRAIN_*=False` class default) | - |
| `README_Instructor.md` | This file: timing, compute profile, licences, citations | - |

The two notebooks are **independent**. Either can be taught first, and either can be dropped if
time is short. Both assume the participants have already attended lectures on CNNs, autoencoders
and diffusion models, and that a transformer lecture follows.

**Class default:** leave `TRAIN_CLEAN`, `TRAIN_SHORTCUT`, and `TRAIN_AUTOENCODER` as `False` so
students load the committed files in `weights/` (downloaded automatically from GitHub if missing).
Set a flag to `True` only when you want a group to reproduce training.

Each notebook marks **CORE** versus **EXTENSION** near the top. In a tight 3-hour session, cut
extensions first (NB1 optional challenges; NB2 bottleneck/harmonization/label-efficiency/UMAP).

### How to distribute

Upload both `.ipynb` files to Google Drive / GitHub / the course page and give participants the
Colab links (`File > Open notebook > GitHub`, or `Upload`). Nothing else is required: no Drive
mount, no Kaggle token, no institutional credentials, no manual downloads, no accounts beyond a
normal Google login. The notebooks ship **without stored outputs** on purpose, so participants see
their own results.

### The pedagogical spine

Both notebooks are built around one message that is easy to state and hard to internalise:

> A model that predicts well is not the same as a model that measures what you think it measures.

Notebook 1 makes this concrete for supervised learning (shortcut learning, attribution maps that
look plausible but prove nothing). Notebook 2 makes it concrete for unsupervised learning
(a latent space that encodes acquisition as faithfully as biology). If participants leave with only
one thing, it should be the habit of asking *what else could explain this result?*

---

# 1. Timing

## Notebook 1 - CNNs and explainability

| Block | Cells | Minutes |
|---|---|---|
| Setup, data download, dataset/reproducibility discussion | 0-4 | 5 |
| Part 1: convolution recap, kernels, receptive fields | 5-8 | 8 |
| Part 2 + Part 3: data inspection, CNN architecture, parameters, tensor shapes, training, predictions | 9-20 | 10 |
| Part 4: feature maps + Exercise 1 | 21-26 | 15 |
| Part 5: Grad-CAM + Exercise 2 | 27-33 | 20 |
| Part 6: perturbation + Exercise 3 | 34-40 | 10 |
| Part 7: simulated shortcut learning + Exercise 4 | 41-48 | 20 |
| Optional challenges, wrap-up, plenary discussion | 49-51 | 10 |
| **Total** | | **~90** |

The two training runs (cells 18 and 43) take 1-2 minutes each on CPU. Start cell 18 and then talk
through the "Think before running" question while it runs.

## Notebook 2 - Autoencoders and latent representations

| Block | Cells | Minutes |
|---|---|---|
| Setup, download, data inspection, Part 1 concepts | 0-7 | 10 |
| Part 2 + Part 3: autoencoder, training, reconstruction, Exercise 1 | 8-15 | 20 |
| Part 4 + Part 5: latent extraction, PCA of latents | 16-20 | 15 |
| Part 6: PCA before vs after autoencoding + Exercise 2 | 21-26 | 10 |
| Part 7: simulated domain shift + Exercise 3 | 27-33 | 20 |
| Part 8: latent interpolation | 34-36 | 10 |
| Part 9: downstream prediction + Exercise 4 | 37-41 | 10 |
| Optional challenges (UMAP), diffusion bridge, wrap-up | 42-46 | 5 |
| **Total** | | **~100** |

Notebook 2 is deliberately slightly over-provisioned. If you are short of time, cut **Exercise 2**
(the `latent_dim=2` retraining, cells 25-26) and the **UMAP** section (cells 43-44). That brings it
to about 80 minutes. Exercise 2 is the only student task in either notebook that trains a model.

### Suggested 3-hour schedule

| Clock | Activity |
|---|---|
| 0:00-0:05 | Framing: the question both notebooks ask |
| 0:05-1:20 | Notebook 1 |
| 1:20-1:30 | Break |
| 1:30-2:50 | Notebook 2 |
| 2:50-3:00 | Joint wrap-up: shortcut learning and batch effects as one phenomenon; bridge to the transformer lecture |

---

# 2. Computational profile

Everything is designed for the **worst case: a CPU-only Colab runtime**. A GPU is used
automatically when present but is never required, and no cell depends on GPU memory.

## Downloads (per participant, whole practical)

| What | Size | Cached where |
|---|---|---|
| `pneumoniamnist_64.npz` (Zenodo, MedMNIST+) | 20.6 MB | `./medmnist_data/` |
| ResNet-18 ImageNet weights (torchvision) | ~45 MB | `~/.cache/torch/hub/checkpoints/` |
| `bloodmnist.npz` (Zenodo, MedMNIST v2) | 35.5 MB | `./medmnist_data/` |
| `umap-learn` wheel (optional section only) | ~5 MB | pip cache |
| **Total** | **~101 MB** (~106 MB with UMAP) | |

Well inside the 1 GB budget. Nothing is written to Google Drive; all files live in the ephemeral
Colab working directory.

## Memory

| Notebook | Largest arrays | Estimated peak RSS |
|---|---|---|
| 1 | all PneumoniaMNIST splits as `uint8` (~24 MB) + ResNet-18 (~45 MB of weights) + one batch of activations | ~2.0-2.5 GB including the PyTorch runtime |
| 2 | 5,600 + 1,600 images as `float32` NCHW (~52 MB + 15 MB), pixel matrix for PCA (~53 MB), latent matrices (<1 MB) | ~2.0-2.5 GB including the PyTorch runtime |

Roughly 1.5 GB of the total is the PyTorch/CUDA runtime itself. Both notebooks stay far below the
8 GB preference and nowhere near the 12 GB limit. GPU memory use, if a GPU is present, is under
500 MB (batch 64 of 1x64x64, or batch 128 of 3x28x28).

## Runtime estimates

The figures below are **estimates for a 2-vCPU Colab CPU runtime**, not measured on your hardware.
Record your own numbers when you run the pre-course check (section 5) and adjust the table.

| Operation | CPU | GPU (T4) |
|---|---|---|
| NB1 dataset download | 5-20 s | same |
| NB1 ResNet-18 weight download | 5-15 s | same |
| NB1 SmallCNN training (2,400 images, 6 epochs, batch 64) | 60-120 s | 10-20 s |
| NB1 feature maps / Grad-CAM cells | 1-5 s each | <1 s |
| NB1 perturbation experiment (60 images) | 20-40 s | <10 s |
| NB1 shortcut-model training (identical size) | 60-120 s | 10-20 s |
| NB2 dataset download | 10-30 s | same |
| NB2 autoencoder training (5,600 images, 20 epochs, batch 128) | 100-240 s | 20-40 s |
| NB2 encoding, PCA, k-NN, logistic regressions | 1-15 s each | same (scikit-learn is CPU-only) |
| NB2 UMAP (optional, two runs) | 20-60 s | same |
| **Total heavy compute** | **~6-10 min** | **<2 GPU-minutes** |

No single training operation exceeds the 3-5 minute CPU guideline. Every model is trained **once**
and cached in `./cached_weights/`: re-running a training cell loads the checkpoint instead of
retraining. To force a retrain, delete the corresponding `.pt` file.

## Packages

All preinstalled in a standard Colab runtime; **no `pip install` is needed for the core path**:

`numpy`, `pandas`, `matplotlib`, `torch`, `torchvision`, `scikit-learn`, `psutil` (optional, for
the RAM report), standard library (`hashlib`, `urllib`, `os`, `time`, `random`, `contextlib`).

Optional only: `umap-learn` (installed inside a `try/except` in the last optional section of
Notebook 2; the section is skipped cleanly if the install fails).

Deliberate non-dependencies: no `medmnist` package (the `.npz` files are downloaded directly from
the official Zenodo record with MD5 verification), no `pytorch-grad-cam` (Grad-CAM is implemented
in ~30 lines so the mechanism is visible), no `torchxrayvision`, no `umap` in the core path.

---

# 3. Data, licences and citations

## PneumoniaMNIST (Notebook 1)

* **Modality**: paediatric chest radiography (frontal), 5,856 images, ages 1-5, Guangzhou Women and Children's Medical Center.
* **Task**: binary classification, `0 = normal`, `1 = pneumonia`.
* **Version used**: 64x64 (`pneumoniamnist_64.npz`), MedMNIST+ release, Zenodo record 10519652, MD5 `8f4eceb4ccffa70c672198ea285246c6`.
* **Splits**: 4,708 / 524 / 624 (train/val/test) as published; the notebook draws class-balanced subsets of 2,400 / 270 / 400 with `SEED = 0`.
* **Licence**: CC BY 4.0 (MedMNIST data); Apache-2.0 (MedMNIST code); source images CC BY 4.0.
* **Not for clinical use.**

## BloodMNIST (Notebook 2)

* **Modality**: light microscopy of individual peripheral blood cells, 17,092 images, CellaVision DM96, Core Laboratory of the Hospital Clinic of Barcelona; donors free of infection, haematologic/oncologic disease and pharmacologic treatment at collection.
* **Task**: 8-class morphological classification (basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil, platelet).
* **Version used**: 28x28 (`bloodmnist.npz`), Zenodo record 10519652, MD5 `7053d0359d879ad8a5505303e11de1dc`.
* **Splits**: 11,959 / 1,712 / 3,421; the notebook draws class-balanced subsets of ~5,600 train and ~1,600 test with `SEED = 0`.
* **Licence**: CC BY 4.0 (MedMNIST data and source dataset); Apache-2.0 (MedMNIST code).
* **Not for clinical use.**

## Data citations (also printed inside both notebooks)

```bibtex
@article{medmnistv2,
  title={MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data}, volume={10}, number={1}, pages={41}, year={2023}}

@inproceedings{medmnistv1,
  title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
  author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
  booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, pages={191--195}, year={2021}}

@article{kermany2018,
  title={Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning},
  author={Kermany, Daniel S. and Goldbaum, Michael and Cai, Wenjia and others},
  journal={Cell}, volume={172}, number={5}, pages={1122--1131}, year={2018}}

@article{acevedo2020,
  title={A dataset of microscopic peripheral blood cell images for development of automatic recognition systems},
  author={Acevedo, Andrea and Merino, Anna and Alferez, Santiago and Molina, Angel and Boldu, Laura and Rodellar, Jose},
  journal={Data in Brief}, volume={30}, pages={105474}, year={2020}}
```

Per MedMNIST's licence terms, users must cite **both** MedMNIST papers **and** the paper of the
source dataset they use. Both notebooks state this.

## Model citations

```bibtex
@inproceedings{he2016resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR}, year={2016}}

@article{russakovsky2015imagenet,
  title={ImageNet Large Scale Visual Recognition Challenge},
  author={Russakovsky, Olga and Deng, Jia and Su, Hao and others},
  journal={IJCV}, volume={115}, number={3}, pages={211--252}, year={2015}}

@inproceedings{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={ICCV}, year={2017}}

@inproceedings{rombach2022ldm,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bjoern},
  booktitle={CVPR}, year={2022}}
```

`torchvision` (BSD-3-Clause) supplies the ResNet-18 architecture and the `IMAGENET1K_V1` weights.
The ImageNet weights are distributed by the PyTorch project for research use; ImageNet itself is
non-commercial research data. No medical pretrained weights are used, and this is discussed
explicitly in Notebook 1 (cell 15 shows the ImageNet head producing meaningless labels for
radiographs).

## Honesty about simulated content

Both notebooks contain synthetic manipulations, each introduced under an explicit **SIMULATED**
heading:

| Notebook | Simulated element | Where |
|---|---|---|
| 1 | 6x6 bright corner marker correlated with the pneumonia label ("Site A annotation habit") | Part 7, cell 42 |
| 1 | occlusion, blur and contrast perturbations | Part 6, cells 35-37 |
| 2 | "Site A" / "Site B" acquisition difference (gamma, channel gain, brightness, noise) | Part 7, cell 28 |

There is **no simulated clinical metadata** anywhere: no fabricated ages, sexes, diagnoses,
scanners or outcomes. When teaching, keep saying "simulated" out loud - participants tend to
remember the finding and forget the caveat.

---

# 4. Pre-course test checklist

Run this **on the morning of the course, in a fresh Colab CPU runtime** (Runtime > Disconnect and
delete runtime first). Zenodo availability, GitHub raw weight URLs, and torchvision ImageNet weights
are the three things that can break without warning.

## Path A — class default (`TRAIN_*=False`)

Confirm the panic-button path students will actually use.

| Notebook | Cell | What to check |
|---|---|---|
| 1 | 3 | prints `TRAIN_CLEAN=False, TRAIN_SHORTCUT=False` |
| 1 | 18 | `loaded instructor weights from weights/cnn_pneumoniamnist.pt` (or GitHub download) |
| 1 | 43 | `loaded instructor weights from weights/cnn_shortcut.pt` |
| 1 | 19, 44, 45 | metrics and Grad-CAM still look sensible with loaded weights |
| 2 | 3 | prints `TRAIN_AUTOENCODER=False` |
| 2 | 9 | `loaded instructor weights from weights/autoencoder_bloodmnist.pt` |
| 2 | 11–12, 28–30 | reconstructions and site AUC still look sensible |

## Path B — retrain once (`TRAIN_*=True`)

Optional but recommended before the course, so you know the fallback numbers if a student flips the flag.

| Cell | What it does | What to check |
|---|---|---|
| NB1 18 | **retrains** clean SmallCNN | 6 epoch lines; note wall-clock time |
| NB1 43 | **retrains** shortcut SmallCNN | high Site-A validation accuracy |
| NB2 9 | **retrains** autoencoder | MSE falling to roughly 0.002–0.006 |

## Shared cells (either path)

| Notebook | Cell | What to check |
|---|---|---|
| 1 | 4 | downloads `pneumoniamnist_64.npz`, MD5 verified |
| 1 | 13 | ResNet-18 ImageNet weights download |
| 1 | 29 | Grad-CAM shapes `(6, 64, 16, 16)` |
| 2 | 4 | downloads `bloodmnist.npz`, MD5 verified |
| 2 | 30 | site-prediction ROC-AUC usually >0.9 |

If cell 30's site-prediction AUC comes out near 0.5, the simulated effect was too weak for the
particular trained autoencoder: increase the effect in `site_b_effect` (for example `gamma=0.75`,
`brightness=0.06`) and re-run cells 28-30. Note the value you get so you can steer the discussion.

## Timing worksheet to fill in

| Measurement | Your value |
|---|---|
| NB1 cell 4 download | |
| NB1 cell 13 ImageNet weight download | |
| NB1 cell 18 instructor-weight load | |
| NB1 cell 18 retrain (optional) | |
| NB1 cell 43 instructor-weight load | |
| NB2 cell 4 download | |
| NB2 cell 9 instructor-weight load | |
| NB2 cell 9 retrain (optional) | |
| NB2 cell 30 site AUC | |

---

# 5. Known failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Download hangs or 5xx error on the `.npz` | Zenodo throttling or downtime | Re-run the cell (it resumes from scratch but the file is small). Fallback: `!pip install medmnist` then `from medmnist import PneumoniaMNIST; PneumoniaMNIST(split="train", size=64, download=True)` / `from medmnist import BloodMNIST; BloodMNIST(split="train", download=True)`, and load the `.npz` from `~/.medmnist/`. |
| `WARNING: MD5 mismatch` | truncated download | Delete `medmnist_data/*.npz` and re-run the download cell. |
| `URLError` on the ResNet-18 weights | network restriction on `download.pytorch.org` | The notebook still works: replace `weights=ResNet18_Weights.IMAGENET1K_V1` with `weights=None` in `build_pretrained_resnet18`. Cells 13-14 and 23 remain meaningful (architecture, shapes, random-filter feature maps); skip cell 15 and mention that the ImageNet demo could not be run. |
| Training much slower than the table | single-vCPU runtime, or a "high-RAM"/shared instance | Reduce `stratified_subset(y_train_all, 1200, RNG)` to `600` and `epochs=6` to `epochs=4` in NB1 cell 18; reduce NB2 to `train_autoencoder(autoencoder, X_tr[:3000], X_te, epochs=10)`. Both keep the pedagogy intact. |
| Model accuracy much lower than expected | unlucky seed, or a subset that is too small after your edits | Change `SEED` in the setup cell, or restore the original subset sizes. |
| Cached weights load but the data settings were changed | stale checkpoint in `cached_weights/` | `!rm -rf cached_weights` and re-run the training cell. |
| `TypeError` on `set_xticks(ticks, labels)` | matplotlib < 3.5 | Replace with `set_xticks(ticks)` followed by `set_xticklabels(labels)`. Colab is well above 3.5. |
| `roc_auc_score` raises about missing classes | a class is absent from the test subset after edits | Keep the balanced subsets, or pass `labels=range(8)`. |
| UMAP import fails after install (numba/llvmlite conflict) | Colab dependency drift | Skip the section; it is optional by design and the notebook handles it in a `try/except`. |
| A Grad-CAM cell raises `element 0 of tensors does not require grad` | the model was wrapped in `torch.no_grad()` | Grad-CAM must run outside `no_grad`; the class already sets `x.requires_grad_(True)`. Do not add `@torch.no_grad()` to a cell that calls `cam_engine`. |
| Kernel restart / disconnect mid-session | Colab idle timeout | Re-run all cells from the top: with `TRAIN_*=False`, recovery is seconds (reload instructor weights). |
| Instructor weight download fails | GitHub raw URL blocked / offline | Place the three `.pt` files from the repo `weights/` folder into a local `weights/` directory in the Colab working directory, or set `TRAIN_*=True` and retrain. |
| Student accidentally set `TRAIN_*=True` and waits | Flag flipped | Tell them to interrupt the cell, set the flag back to `False`, Runtime > Restart session, and re-run from the top. |

---

# 6. Teaching notes

## Notebook 1

* **Cell 8 (receptive fields).** The point that lands hardest: for a 64x64 image, ResNet-18's deep
  units see the entire image. Ask "so what could a deep unit be responding to?" before revealing
  Part 7.
* **Cell 15 (ImageNet labels on X-rays).** Reliably gets a laugh, and it sets up the distinction
  between transferable *features* and non-transferable *heads*.
* **Cells 22-23 (feature maps).** Resist interpreting individual channels. If a participant says
  "channel 5 detects the lungs", ask them to check whether it also fires on the image border.
* **Cell 29-30 (Grad-CAM).** The two most useful teaching moves: (i) show that the pneumonia map
  and the normal map are strongly correlated, which surprises people; (ii) show the 2x2 ResNet-18
  map to make "resolution is a design choice" concrete.
* **Cell 37 (perturbation with control).** Emphasise the control condition. Many published
  perturbation analyses omit it, which makes the result uninterpretable.
* **Cells 42-45 (shortcut).** The pedagogical climax. Keep the three test sets on the board:
  internal, clean, swapped. Then ask which one a typical paper reports.
* **Plenary prompt.** "Zech et al. (2018) found a model that identified the hospital from the
  radiograph. What is the marker in that story?" (Portable-scanner tokens, laterality markers,
  collimation patterns.)

Common misconceptions to correct explicitly:

1. "Grad-CAM shows what the model sees." It shows where the gradient of one scalar is large in one
   layer. Nothing more.
2. "High validation accuracy plus a plausible heat map equals a valid model."
3. "Shortcut learning happens because the model is too big." It happens because the shortcut is
   predictive and cheap to extract; capacity is not the mechanism.
4. "Data augmentation solves it." Only if the augmentation destroys the shortcut, which requires
   knowing what the shortcut is.

## Notebook 2

* **Cell 9 (training).** Say "demonstration scale" out loud. Participants routinely assume a
  three-minute autoencoder is a research artefact.
* **Cells 11-12 (reconstruction).** Push on the MSE argument: a small nuclear inclusion covering
  1% of the pixels can be discarded almost for free. This is why reconstruction quality is a poor
  proxy for clinical adequacy.
* **Cell 19 (latent PCA).** Ask "the model never saw a label, so why do the classes separate?"
  Answer: cell type is a dominant source of pixel variance, so it is cheap to encode. Then ask what
  would happen for a phenotype that is *not* a dominant source of variance (subtle dysplasia,
  early disease) - it would be discarded.
* **Cells 28-30 (site effect).** The single most transferable result of the whole practical. Get the
  site-prediction AUC on the board next to the cell-type accuracy.
* **Cell 35 (interpolation).** Ask for a show of hands: "who thinks the midpoint is a real cell?"
  Then decode the walk between two neutrophils to show that smoothness is a property of the
  decoder, not of biology.
* **Cell 38 (linear probe).** Connect to foundation models and the label-scarcity economics of
  health research, which is the practical reason clinicians care about representation learning.
* **Cells 45-46 (diffusion bridge).** Two sentences are enough: latent diffusion works in the space
  you just built, and it inherits both the bottleneck's blind spots and the archive's batch
  structure.

Common misconceptions to correct explicitly:

1. "Unsupervised means unbiased." The objective is pixel variance, and technical variation is
   variance.
2. "Good reconstruction means good representation."
3. "Clusters in a 2D embedding are subtypes."
4. "Smooth interpolation means a disease trajectory."
5. "Batch correction removes the problem." It removes a *marginal* difference under assumptions
   that fail when site and biology are associated.

---

# 7. Exercise solutions

Drop-in exercise code and discussion keys are **not** published in this repository.
Keep them in the local instructor answer key (`TEACHER_ANSWERS.md`).

---

# 8. Adaptations

| Situation | What to change |
|---|---|
| Only 2 hours available | Notebook 1: keep Parts 1-5 and 7, drop Part 6 and Exercise 3. Notebook 2: keep Parts 1-5 and 7 and 9, drop Exercise 2 and UMAP. |
| Participants are slower than expected | Assign Exercises 1 and 3 (NB1) and Exercise 1 (NB2) as homework; run the guided cells together. |
| GPU available for everyone | Nothing needs to change. Optionally raise the subsets (`1200 -> 3000` per class in NB1, `700 -> 1500` in NB2) and epochs; retrain times stay under a minute. |
| You want stored outputs for a reference copy | Run both notebooks once end to end in Colab and save a copy as `*_solved.ipynb`; keep the distributed versions output-free. |
| Fully offline teaching room | Pre-download `pneumoniamnist_64.npz`, `bloodmnist.npz` and the ResNet-18 checkpoint, distribute them, and replace the `download(...)` calls with local paths (`resnet18` then needs `weights=None` plus `load_state_dict`). |
| Assessment | The four interpretation blocks of each notebook work well as a short written assignment; the shortcut-learning limitations paragraph (NB1 Exercise 4) is the single best assessment item. |

## Provenance of this material

The notebooks were authored for this course and use only public data and public model weights. They
are distributed with no stored outputs, and the compute figures in section 2 are engineering
estimates for a 2-vCPU Colab CPU runtime rather than measurements - fill in the worksheet in
section 4 with your own numbers before teaching.



