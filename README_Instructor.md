# Instructor Guide

## Advanced Data Science and Machine Learning for Health Research - 3-hour practical

| File | Content | Duration |
|---|---|---|
| `01_CNN_Explainability_Health.ipynb` | Practical 1: CNNs and explainability in medical imaging (PneumoniaMNIST) | 80-90 min |
| `02_Autoencoder_Latent_Space_Health.ipynb` | Practical 2: *What Is Hidden in a Learned Representation?* (BloodMNIST) | 80-85 min |
| `weights/` | Instructor fallback checkpoints (`TRAIN_*=False` class default) | - |
| `README_Instructor.md` | This file: timing, compute profile, licences, citations, full solutions | - |

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

# 7. Complete solutions - Notebook 1

Each block is a drop-in replacement for the corresponding exercise cell and uses only names that
already exist in the notebook.

## Exercise 1 - compare early and late feature maps

```python
# ===== EXERCISE 1 - SOLUTION =====
i_normal = int(np.flatnonzero(y_te == 0)[0])
i_pneumo = int(np.flatnonzero(y_te == 1)[0])

pair = X_te[[i_normal, i_pneumo]]
acts = get_activations(cnn, prep_gray(pair), ["block1", "block3"])

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
for row, layer in enumerate(["block1", "block3"]):
    m_norm = acts[layer][0].mean(0).numpy()
    m_pneu = acts[layer][1].mean(0).numpy()
    abs_diff = np.abs(m_norm - m_pneu)
    rel_diff = abs_diff.mean() / (acts[layer].mean().item() + 1e-8)

    axes[row, 0].imshow(m_norm, cmap="magma")
    axes[row, 0].set_title(f"{layer} - normal\n{m_norm.shape}", fontsize=8)
    axes[row, 1].imshow(m_pneu, cmap="magma")
    axes[row, 1].set_title(f"{layer} - pneumonia", fontsize=8)
    axes[row, 2].imshow(abs_diff, cmap="inferno")
    axes[row, 2].set_title(f"|difference|\nmean {abs_diff.mean():.3f}, "
                           f"relative {rel_diff:.3f}", fontsize=8)
    print(f"{layer:7s}: mean|diff| = {abs_diff.mean():.4f}, "
          f"layer mean activation = {acts[layer].mean().item():.4f}, "
          f"relative difference = {rel_diff:.3f}")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()
```

**Expected result and model answer.** The absolute difference is usually *larger in absolute terms*
in `block1` (early maps have high, image-like magnitudes everywhere), but the *relative* difference
is larger in `block3`. Early layers respond to edges present in both images, so the two classes look
similar there; the deep layer is where class-relevant configurations are encoded, and it is also
where spatial detail has been discarded. This matches the hierarchy in Part 1: early = generic and
shared, deep = specific and coarse. Caution the participants that a single image pair is anecdotal;
the honest version of this exercise averages over many images per class.

## Exercise 2 - Grad-CAM on your own selection

```python
# ===== EXERCISE 2 - SOLUTION =====
p_wrong = probs_te[np.arange(len(y_te)), 1 - y_te]     # probability of the WRONG class
p_right = probs_te[np.arange(len(y_te)), y_te]
i_worst = int(p_wrong.argmax())
i_best = int(p_right.argmax())
print(f"most confidently wrong: index {i_worst}, true {CLASS_NAMES[y_te[i_worst]]}, "
      f"P(wrong class) = {p_wrong[i_worst]:.3f}")
print(f"most confidently right: index {i_best}, true {CLASS_NAMES[y_te[i_best]]}, "
      f"P(true class) = {p_right[i_best]:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))
for row, (i, tag) in enumerate([(i_worst, "confidently WRONG"), (i_best, "confidently right")]):
    pred_class = int(probs_te[i].argmax())
    cam_pred, _ = cam_engine(prep_gray(X_te[i]), class_idx=pred_class)
    cam_true, _ = cam_engine(prep_gray(X_te[i]), class_idx=int(y_te[i]))
    axes[row, 0].imshow(X_te[i], cmap="gray"); axes[row, 0].axis("off")
    axes[row, 0].set_title(f"{tag}\ntrue {CLASS_NAMES[y_te[i]]}", fontsize=8)
    overlay_cam(axes[row, 1], X_te[i], cam_pred[0],
                title=f"CAM, predicted = {CLASS_NAMES[pred_class]}")
    overlay_cam(axes[row, 2], X_te[i], cam_true[0],
                title=f"CAM, true = {CLASS_NAMES[y_te[i]]}")
plt.tight_layout(); plt.show()
```

**Model answer.** The two maps for a misclassified image are usually very similar, so the
attribution map does *not* explain the error: it tells you which positions influenced a score, not
why the score was wrong. Useful follow-up question for the room: what would explain the error?
Candidates include an atypical presentation, a labelling error in the source data, an unusual
exposure or field of view, and the fact that the model has 24k parameters and no access to clinical
context. This is the moment to say that attribution maps are diagnostics of the model, not error
analyses.

## Exercise 3 - quantify attribution with occlusion

```python
# ===== EXERCISE 3 - SOLUTION =====
idx = np.argsort(-probs_te[:, 1])[:20]                  # 20 most confident pneumonia predictions
deltas = []
for i in idx:
    base = float(probs_te[i, 1])
    cam_i, _ = cam_engine(prep_gray(X_te[i]), class_idx=1)
    top, left = top_cam_region(cam_i[0], size=16)
    deltas.append(p_pneumonia(cnn, occlude(X_te[i], top, left, 16)) - base)
deltas = np.array(deltas)

print(f"mean change in P(pneumonia): {deltas.mean():+.3f}")
print(f"10th / 90th percentile      : {np.percentile(deltas, 10):+.3f} / "
      f"{np.percentile(deltas, 90):+.3f}")
print(f"probability fell in {(deltas < 0).mean()*100:.0f}% of the 20 images")

most_sensitive = idx[np.argsort(np.abs(deltas))[-2:]]
fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.2))
for row, i in enumerate(most_sensitive):
    cam_i, _ = cam_engine(prep_gray(X_te[i]), class_idx=1)
    top, left = top_cam_region(cam_i[0], size=16)
    overlay_cam(axes[row, 0], X_te[i], cam_i[0], title=f"Grad-CAM (index {i})")
    axes[row, 1].imshow(occlude(X_te[i], top, left, 16), cmap="gray", vmin=0, vmax=255)
    axes[row, 1].axis("off")
    axes[row, 1].set_title(f"occluded, P = "
                           f"{p_pneumonia(cnn, occlude(X_te[i], top, left, 16)):.3f}", fontsize=8)
plt.tight_layout(); plt.show()
```

**Model answer to "is a large delta evidence that the model is right?"** No. A large delta shows
only that the model's output depends on that region - it is a statement about sensitivity, not
correctness. The region could contain a lesion, or a rib, or an annotation, or the image border.
Additionally, occlusion inserts an out-of-distribution grey square, so part of the change may be a
response to the artefact rather than to the removal of information. Evidence of correctness requires
external data, a comparison with an independent reference standard (for example radiologist
annotations), or an experiment in which the biological signal is manipulated while the nuisance is
held constant.

## Exercise 4 - identify and quantify the shortcut

```python
# ===== EXERCISE 4 - SOLUTION =====
idx_marked_pneu = np.flatnonzero((y_te == 1) & m_teA)      # pneumonia images that carry a marker
idx_clean_normal = np.flatnonzero((y_te == 0) & ~m_teA)    # normal images without a marker
print(f"n marked pneumonia = {len(idx_marked_pneu)}, n unmarked normal = {len(idx_clean_normal)}")

rows = []
for name, model in [("shortcut model", cnn_shortcut), ("clean model", cnn)]:
    with_marker = predict_probs(model, X_te_siteA[idx_marked_pneu])[:, 1].mean()
    marker_removed = predict_probs(model, X_te[idx_marked_pneu])[:, 1].mean()
    without = predict_probs(model, X_te[idx_clean_normal])[:, 1].mean()
    marker_added = predict_probs(model, add_marker(X_te[idx_clean_normal]))[:, 1].mean()
    rows.append((name, with_marker, marker_removed, marker_removed - with_marker,
                 without, marker_added, marker_added - without))

print(f"\n{'model':16s} {'pneu +mk':>9s} {'pneu -mk':>9s} {'delta':>8s} "
      f"{'norm -mk':>9s} {'norm +mk':>9s} {'delta':>8s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:>9.3f} {r[2]:>9.3f} {r[3]:>+8.3f} "
          f"{r[4]:>9.3f} {r[5]:>9.3f} {r[6]:>+8.3f}")
```

**Expected pattern.** For the shortcut model, removing the marker from pneumonia images drops
P(pneumonia) substantially (often by 0.2-0.6), and adding a marker to normal images raises it by a
similar amount. For the clean model both deltas are small (typically |delta| < 0.05), because it
never saw the marker during training. The 6x6 marker occupies 0.9% of the pixels.

**Model limitations paragraph (example).** *We trained a convolutional classifier to detect
pneumonia on paediatric chest radiographs from a single simulated site and obtained high internal
validation accuracy. A post-hoc analysis showed that predictions depend strongly on a small
annotation-like marker in the upper-left corner of the image: removing it from positive cases
lowered the predicted probability by [X], and adding it to negative cases raised it by [Y]. Grad-CAM
confirmed that attribution concentrated on this region rather than on lung parenchyma. Because the
marker was present in 95% of positive and 5% of negative training images, it was a stronger and
cheaper predictor than the radiographic signs of consolidation, and gradient descent exploited it.
Internal validation could not detect this failure, since the validation set was drawn from the same
distribution and therefore contained the same marker-label association: the internal estimate is an
unbiased estimate of performance in that distribution, not of performance under a different
annotation convention. Accuracy fell substantially on a marker-free test set and further on a set
where the annotation habit was reversed, demonstrating that the reported accuracy is not
transportable. Reported performance should therefore be interpreted as site-specific, and external
validation on data from centres with different acquisition and annotation practices is required
before any claim of generalisability.*

## Optional challenges - Notebook 1

```python
# 1) How strong must the shortcut be?
X_tr_weak, m_tr_weak = inject_shortcut(X_tr, y_tr, p_marker_pos=0.65, p_marker_neg=0.35,
                                       rng=np.random.default_rng(11))
X_va_weak, _ = inject_shortcut(X_va, y_va, p_marker_pos=0.65, p_marker_neg=0.35,
                               rng=np.random.default_rng(12))
cnn_weak = SmallCNN().to(DEVICE)
with timed("train weak-shortcut model"):
    fit_or_load(cnn_weak, "smallcnn_shortcut_weak.pt",
                lambda: train_classifier(cnn_weak, X_tr_weak, y_tr, X_va_weak, y_va, epochs=6))
for tname, X_set in test_sets.items():
    p = predict_probs(cnn_weak, X_set)
    print(f"{tname.replace(chr(10), ' '):24s} accuracy {accuracy_score(y_te, p.argmax(1)):.3f}")
# Expect a much smaller internal-to-clean gap: with a weak association the marker is worth less
# than the real signal, so the model relies on it less. There is no sharp threshold - reliance
# scales with how predictive and how easy to extract the shortcut is.

# 2) Grad-CAM sanity check on an untrained network
random_net = SmallCNN().to(DEVICE)
cam_random = GradCAM(random_net, random_net.block3)
cams_r, _ = cam_random(prep_gray(X_te[sel[:4]]), class_idx=1)
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for col in range(4):
    overlay_cam(axes[col], X_te[sel[col]], cams_r[col], title="random weights")
plt.tight_layout(); plt.show()
cam_random.close()
# The maps are smooth, structured and completely meaningless: an untrained network still has
# spatially varying activations and gradients. Structure in a heat map is not evidence of learning.
# This is the intuition behind Adebayo et al. (2018), "Sanity checks for saliency maps".

# 3) Occlusion sensitivity map
i = int(sel[0])
size, stride = 12, 4
positions = list(range(0, 64 - size + 1, stride))
sens = np.zeros((len(positions), len(positions)))
base = p_pneumonia(cnn, X_te[i])
for a, top in enumerate(positions):
    for b, left in enumerate(positions):
        sens[a, b] = base - p_pneumonia(cnn, occlude(X_te[i], top, left, size))
sens_up = F.interpolate(torch.from_numpy(sens).float().view(1, 1, *sens.shape),
                        size=(64, 64), mode="bilinear", align_corners=False)[0, 0].numpy()
cam_i, _ = cam_engine(prep_gray(X_te[i]), class_idx=1)
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
axes[0].imshow(X_te[i], cmap="gray"); axes[0].axis("off"); axes[0].set_title("input", fontsize=8)
overlay_cam(axes[1], X_te[i], cam_i[0], title="Grad-CAM (1 backward pass)")
overlay_cam(axes[2], X_te[i], (sens_up - sens_up.min()) / (np.ptp(sens_up) + 1e-8),
            title=f"occlusion map ({sens.size} forward passes)")
plt.tight_layout(); plt.show()
# Occlusion measures the model's behaviour directly and needs no gradients, but costs one forward
# pass per position and depends on the choice of occluder. Grad-CAM costs one backward pass but is
# a first-order approximation restricted to one layer's resolution.

# 4) Frozen-feature transfer-learning probe
from sklearn.linear_model import LogisticRegression
feature_extractor = nn.Sequential(*(list(resnet.children())[:-1])).eval().to(DEVICE)
with torch.no_grad():
    F_tr = feature_extractor(prep_rgb_imagenet(X_tr[:800])).flatten(1).cpu().numpy()
    F_te = feature_extractor(prep_rgb_imagenet(X_te)).flatten(1).cpu().numpy()
probe = LogisticRegression(max_iter=2000).fit(F_tr, y_tr[:800])
p_probe = probe.predict_proba(F_te)[:, 1]
print(f"ImageNet features + logistic regression: accuracy "
      f"{accuracy_score(y_te, (p_probe > 0.5).astype(int)):.3f}, "
      f"ROC-AUC {roc_auc_score(y_te, p_probe):.3f}")
print(f"SmallCNN trained from scratch          : accuracy "
      f"{accuracy_score(y_te, pred_te):.3f}, ROC-AUC {roc_auc_score(y_te, probs_te[:, 1]):.3f}")
# Frozen ImageNet features usually get within a few points of the task-trained SmallCNN with 800
# labels and no backpropagation through the backbone: a useful demonstration of why transfer
# learning dominates small-sample medical imaging work.
```

---

# 8. Complete solutions - Notebook 2

## Exercise 1 - reconstruction error as a quality-control statistic

```python
# ===== EXERCISE 1 - SOLUTION =====
lo, hi = np.quantile(errors_te, [0.05, 0.95])
idx_low = np.flatnonzero(errors_te <= lo)
idx_high = np.flatnonzero(errors_te >= hi)
print(f"5% easiest  (MSE <= {lo:.4f}): n={len(idx_low)}")
print(f"5% hardest  (MSE >= {hi:.4f}): n={len(idx_high)}")

comp = pd.DataFrame({"easiest 5%": np.bincount(y_te[idx_low], minlength=8),
                     "hardest 5%": np.bincount(y_te[idx_high], minlength=8),
                     "whole test set": np.bincount(y_te, minlength=8)}, index=CLASS_NAMES)
print(comp.to_string())

fig, axes = plt.subplots(2, 4, figsize=(7, 3.8))
for col in range(4):
    axes[0, col].imshow(to_image(X_te[idx_low[col]]))
    axes[0, col].set_title(f"easy\n{CLASS_NAMES[y_te[idx_low[col]]][:9]}", fontsize=7)
    axes[1, col].imshow(to_image(X_te[idx_high[col]]))
    axes[1, col].set_title(f"hard\n{CLASS_NAMES[y_te[idx_high[col]]][:9]}", fontsize=7)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()

X_bright = (X_te * 1.25).clamp(0, 1)
err_bright = reconstruction_errors(autoencoder, X_bright)
print(f"\nmean reconstruction MSE, original images : {errors_te.mean():.5f}")
print(f"mean reconstruction MSE, brightened images: {err_bright.mean():.5f} "
      f"({err_bright.mean()/errors_te.mean():.1f}x higher)")
```

**Expected result and model answer.** The easiest tail is dominated by small, homogeneous, mostly
background images (platelets are typical); the hardest tail is dominated by large, textured cells
(immature granulocytes, monocytes, eosinophils). A purely global brightness change raises the mean
error noticeably, even though no biological information changed. Consequence: reconstruction error
is not a pure "abnormality" score, it is an "unusual for the training distribution" score, and it
confounds technical deviation with biological rarity. Using it as a QC filter would preferentially
exclude images from centres whose acquisition differs from the training centre *and* the rare cell
types you probably care most about, which introduces selection bias in both the exposure and the
outcome direction.

## Exercise 2 - the bottleneck controls what is learned

```python
# ===== EXERCISE 2 - SOLUTION =====
ae2 = ConvAutoencoder(latent_dim=2).to(DEVICE)
with timed("train latent_dim=2 autoencoder"):
    fit_or_load(ae2, "conv_autoencoder_z2.pt",
                lambda: train_autoencoder(ae2, X_tr[:3000], X_te, epochs=10))

X_te_hat2 = reconstruct(ae2, X_te)
err2 = reconstruction_errors(ae2, X_te)
print(f"mean test MSE: latent_dim=32 -> {errors_te.mean():.5f}, "
      f"latent_dim=2 -> {err2.mean():.5f}")

fig, axes = plt.subplots(3, 8, figsize=(11, 4.4))
for col, i in enumerate(show):
    axes[0, col].imshow(to_image(X_te[i])); axes[0, col].set_title(CLASS_NAMES[y_te[i]][:9],
                                                                  fontsize=7)
    axes[1, col].imshow(to_image(X_te_hat[i]))
    axes[2, col].imshow(to_image(X_te_hat2[i]))
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("Row 1 original | Row 2 reconstruction with z=32 | Row 3 with z=2", fontsize=10)
plt.tight_layout(); plt.show()

Z2_tr, Z2_te = encode_all(ae2, X_tr), encode_all(ae2, X_te)
plt.figure(figsize=(5, 4))
for c in range(8):
    m = y_te == c
    plt.scatter(Z2_te[m, 0], Z2_te[m, 1], s=8, alpha=0.65, color=cmap(c), label=CLASS_NAMES[c])
plt.xlabel("z1"); plt.ylabel("z2"); plt.legend(fontsize=6, markerscale=1.4)
plt.title("The entire latent space, latent_dim = 2")
plt.tight_layout(); plt.show()

print(f"10-NN accuracy: z=2 -> {knn_accuracy(Z2_tr, Z2_te):.3f}, "
      f"z=32 -> {acc_latent:.3f}, 32 pixel PCs -> {acc_pix_pca:.3f}")
```

**Model answer.** With two latent dimensions the reconstructions collapse towards class-average
blobs: colour and overall size survive, all internal structure is gone, and reconstruction MSE rises
substantially. The 2D latent space is directly plottable and still separates several cell types
above chance, but *k*-NN accuracy drops clearly relative to `z=32`. The trade-off: a tighter
bottleneck buys interpretability and a directly visualisable space at the cost of discarding
information, and the information it discards is chosen by the *pixel-variance* objective rather than
by clinical relevance. In a paper, report the downstream metric (accuracy on the task you care
about) as primary, not the reconstruction loss - and report the latent dimension as the design
choice it is.

## Exercise 3 - harmonise the simulated site effect

```python
# ===== EXERCISE 3 - SOLUTION =====
Z_harm = Z_pool.copy()
for s in (0, 1):
    m = site == s
    Z_harm[m] = (Z_pool[m] - Z_pool[m].mean(0)) / (Z_pool[m].std(0) + 1e-8)

# (2) can we still predict the site?
Ztr_h, Zte_h, s_tr_h, s_te_h = train_test_split(Z_harm, site, test_size=0.4,
                                                random_state=SEED, stratify=site)
sc_h = StandardScaler().fit(Ztr_h)
clf_h = LogisticRegression(max_iter=2000).fit(sc_h.transform(Ztr_h), s_tr_h)
auc_h = roc_auc_score(s_te_h, clf_h.predict_proba(sc_h.transform(Zte_h))[:, 1])
print(f"site prediction ROC-AUC: raw latents {roc_auc_score(s_te, s_prob):.3f} "
      f"-> harmonised {auc_h:.3f}")

# (3) does transportability improve?
sc_c = StandardScaler().fit(Z_harm[a_train])
clf_c = LogisticRegression(max_iter=3000).fit(sc_c.transform(Z_harm[a_train]), y_pool[a_train])
within_h = balanced_accuracy_score(y_pool[a_test], clf_c.predict(sc_c.transform(Z_harm[a_test])))
across_h = balanced_accuracy_score(y_pool[idx_b], clf_c.predict(sc_c.transform(Z_harm[idx_b])))
print(f"cell-type balanced accuracy, harmonised: Site A {within_h:.3f}, Site B {across_h:.3f} "
      f"(before: {acc_within:.3f} / {acc_across:.3f})")
```

**Expected result.** Per-site standardisation typically pushes the site AUC substantially towards
0.5 (it removes exactly the first two moments of the site difference) and recovers part - rarely all
- of the cross-site accuracy, because the residual site effect is nonlinear and interacts with
class.

**Model answer to the interpretation question.** Step 1 assumes that the *distribution of biology is
the same in both sites*, so that any difference in latent means and variances is attributable to
acquisition. In this simulation that assumption is true by construction: we assigned site at
random. In a real multicentre study it is almost never true - centres differ in case mix - and then
per-site centring removes real biological signal along with the nuisance. In the extreme, if Site B
genuinely sees more immature granulocytes, standardising within site erases exactly the difference
you wanted to measure, biasing any comparison of sites towards the null. This is the same trap as
over-adjustment in epidemiology, and the same criticism made of naive batch correction when batch is
correlated with the outcome. Safer alternatives: harmonise at the image level before encoding
(so that the correction cannot see the labels), include site as a covariate rather than removing it,
or train the representation with augmentation that makes it invariant to plausible acquisition
changes - and in all cases report the site-predictability of the final representation as a
diagnostic.

## Exercise 4 - how many labels do you actually need?

```python
# ===== EXERCISE 4 - SOLUTION =====
P_pix_tr = pca_pixels.transform(pixels_tr)
sizes = [50, 100, 250, 500, 1000, len(Z_tr)]
curves = {"autoencoder latents (32)": [], "pixel PCA (32)": []}

for n in sizes:
    if n >= len(Z_tr):
        idx = np.arange(len(Z_tr))
    else:
        idx, _ = train_test_split(np.arange(len(Z_tr)), train_size=n,
                                  random_state=SEED, stratify=y_tr)
    for name, F_tr in [("autoencoder latents (32)", Z_tr), ("pixel PCA (32)", P_pix_tr)]:
        F_te = Z_te if name.startswith("autoencoder") else P_pix_te
        scaler = StandardScaler().fit(F_tr[idx])
        clf = LogisticRegression(max_iter=3000).fit(scaler.transform(F_tr[idx]), y_tr[idx])
        curves[name].append(balanced_accuracy_score(y_te, clf.predict(scaler.transform(F_te))))

plt.figure(figsize=(5.2, 3.2))
for name, values in curves.items():
    plt.semilogx(sizes, values, marker="o", label=name)
plt.axhline(0.125, color="grey", ls="--", lw=0.8, label="chance")
plt.xlabel("number of labelled training images"); plt.ylabel("test balanced accuracy")
plt.legend(fontsize=7); plt.title("Label efficiency")
plt.tight_layout(); plt.show()

for name, values in curves.items():
    print(f"{name:28s} " + "  ".join(f"n={n}: {v:.3f}" for n, v in zip(sizes, values)))
```

**Model answer.** Both curves rise with `n` and flatten; the autoencoder latents are usually ahead
at small `n` and the gap narrows as labels accumulate. The practical reading for a study with about
200 expert-labelled images: the representation matters most exactly in that regime, which is the
regime most clinical research operates in. Two caveats worth stating: (i) the autoencoder was
trained on the *same* images (unlabelled), which is legitimate transductive use of unlabelled data
but must be described honestly; (ii) with 50-100 labels the confidence intervals are wide, so a
single split can easily reverse the ordering - a proper version of this experiment repeats each
sample size over several random draws and reports the spread.

## Optional challenges - Notebook 2

```python
# 1) Denoising autoencoder: same loop, but the input is corrupted and the target stays clean
def train_denoising(model, X, X_val=None, epochs=10, batch=128, lr=2e-3, sigma=0.15, seed=SEED):
    model.to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    n = len(X)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.from_numpy(np.random.default_rng(seed + epoch).permutation(n))
        running = 0.0
        for start in range(0, n, batch):
            xb = X[order[start:start + batch]].to(DEVICE)
            noisy = (xb + sigma * torch.randn_like(xb)).clamp(0, 1)
            optimiser.zero_grad(set_to_none=True)
            x_hat, _ = model(noisy)
            loss = F.mse_loss(x_hat, xb)          # target is the CLEAN image
            loss.backward(); optimiser.step()
            running += loss.item() * len(xb)
        history["epoch"].append(epoch)
        history["train_loss"].append(running / n)
        history["val_loss"].append(float(np.mean(reconstruction_errors(model, X_val)))
                                   if X_val is not None else float("nan"))
        print(f"epoch {epoch}/{epochs}  train MSE {running / n:.5f}")
    return history

dae = ConvAutoencoder(latent_dim=32).to(DEVICE)
with timed("train denoising autoencoder"):
    fit_or_load(dae, "conv_dae_z32.pt",
                lambda: train_denoising(dae, X_tr[:3000], X_te, epochs=10))
Z_pool_dae = encode_all(dae, X_pool)
Ztr_d, Zte_d, s_tr_d, s_te_d = train_test_split(Z_pool_dae, site, test_size=0.4,
                                                random_state=SEED, stratify=site)
sc_d = StandardScaler().fit(Ztr_d)
clf_d = LogisticRegression(max_iter=2000).fit(sc_d.transform(Ztr_d), s_tr_d)
print(f"site AUC, plain autoencoder {roc_auc_score(s_te, s_prob):.3f} vs denoising "
      f"{roc_auc_score(s_te_d, clf_d.predict_proba(sc_d.transform(Zte_d))[:, 1]):.3f}")
# Denoising makes the representation invariant to the noise it was trained against - not to gamma
# or colour shifts. Expect the site AUC to stay high: invariance must be built against the specific
# nuisance you care about, which is the core idea behind augmentation-based domain generalisation.

# 2) Latent traversal
i = int(np.flatnonzero(y_te == 6)[0])
z0 = torch.from_numpy(Z_te[i]).float()
sds = Z_te.std(0)
dims = [0, 1, 2, 3]
deltas = np.linspace(-3, 3, 7)
fig, axes = plt.subplots(len(dims), len(deltas), figsize=(1.1 * len(deltas), 1.15 * len(dims)))
for r, j in enumerate(dims):
    for c, d in enumerate(deltas):
        z = z0.clone(); z[j] = z0[j] + d * sds[j]
        with torch.no_grad():
            img = autoencoder.decode(z.view(1, -1).to(DEVICE))[0].cpu()
        axes[r, c].imshow(to_image(img)); axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(f"{d:+.0f} sd", fontsize=7)
    axes[r, 0].set_ylabel(f"z{j}", fontsize=7)
plt.tight_layout(); plt.show()
# Some dimensions visibly control brightness, size or hue; most control entangled mixtures. Plain
# autoencoders have no disentanglement pressure, so "one dimension = one biological factor" is not
# expected and should never be assumed.

# 3) Clustering without labels
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
km_clean = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit_predict(Z_te)
km_pool = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit_predict(Z_pool)
print(f"ARI vs true cell type, single site  : {adjusted_rand_score(y_te, km_clean):.3f}")
print(f"ARI vs true cell type, two sites    : {adjusted_rand_score(y_pool, km_pool):.3f}")
print(f"ARI vs SITE label, two sites        : {adjusted_rand_score(site, km_pool):.3f}")
# The third number is the interesting one: some clusters track the simulated site rather than
# biology. This is exactly how "novel subtypes" can turn out to be scanners.

# 4) Variational bottleneck - expected answer
# A VAE adds KL(q(z|x) || N(0, I)) to the reconstruction loss. This forces the aggregate posterior
# towards a known prior, so points sampled from N(0, I) land in regions the decoder has been trained
# to map to plausible images. A plain autoencoder has no such constraint: its latent codes occupy an
# arbitrary, possibly disconnected region, so sampling from a standard normal decodes to nonsense.
# Latent diffusion needs a well-behaved, roughly isotropic latent space to noise and denoise, which
# is why its first stage is a (KL- or VQ-regularised) autoencoder rather than a plain one.
```

---

# 9. Answer keys for the discussion questions

Short model answers. They are deliberately compressed; the value is in the discussion, not the
wording.

## Notebook 1

**Part 2 interpretation (looking at the raw images).** Most participants can separate the classes
only weakly; typical cues are diffuse opacity and loss of sharp vascular markings. Non-diagnostic
differences that are visible even at 64x64: field of view, rotation, how much abdomen is included,
overall exposure, and the amount of black border. Alternative explanations for 95% accuracy:
exposure/contrast differences between the wards where the two groups were imaged, patient
positioning or age-related body size, and label noise correlated with acquisition.

**Part 3 think-before-running.** ResNet-18's final layer has 512 x 1000 + 1000 = 513,000 parameters,
about 4.4% of the model; the capacity is overwhelmingly convolutional. The ImageNet head predicts
irrelevant object classes for radiographs.

**Part 4 interpretation.** (1) Depth reduces spatial resolution and increases abstraction: `block1`
maps still look like a chest, `block3` maps are sparse and unrecognisable. (2) Early features are
generic (edges, gradients), which is why an ImageNet model's `conv1` responds sensibly to a
radiograph - and why transfer learning works at all. (3) No: channels are not identifiable
(permutations and rescalings give the same function), single channels are polysemantic, apparent
selectivity is usually tested on far too few images, and correlation with anatomy is not evidence of
mechanism. (4) Because position in a radiograph is confounded with everything that is
systematically located there: the right lower lobe region also contains a particular rib pattern,
the diaphragm edge, and in many archives the area where support devices or annotations appear.

**Part 5 interpretation.** (1) Both logits are linear functions of the *same* globally pooled
channels, so their gradients are near mirror images and the ReLU-ed maps overlap heavily; the two
maps are highly correlated by construction. (2) `block1` maps look sharper but are less
class-specific; `block3` maps are class-specific but coarse. Resolution and semantic level trade off
because both come from the same downsampling. (3) No: plausibility is not validity. The map shows
which positions in one layer increased one score. Additional evidence would be perturbation with
proper controls, agreement with independent expert annotations, stability across seeds and
architectures, and above all external validation.

**Part 6 interpretation.** (1) Usually yes on average, but not for every image; report the paired
difference, not just the mean. (2) Occlusion can increase the probability when the occluded region
contained evidence *against* pneumonia, or when the grey square itself resembles an opacity. (3) The
grey square is out of distribution, so part of the effect is a response to an artefact; more
realistic perturbations are local blurring, in-painting with surrounding texture, or replacing the
region with the same region from another patient. (4) Any non-trivial change in probability from a
global contrast rescaling means the model is sensitive to exposure, which differs by protocol and
vendor - a direct threat to transportability.

**Part 7 interpretation.** (1) The internal estimate is an unbiased estimate of accuracy *in the
distribution it was drawn from*, including that distribution's marker-label association. It is not
an estimate of accuracy anywhere else. (2) No; the clean test set is the analogue of a hospital
without that annotation habit, and the swapped set of a hospital with the opposite habit.
(3) Exposure = the image (or the model's use of it), outcome = pneumonia, confounder = the
annotation habit, which is associated with the outcome (through triage/ward) and affects the image.
The model's use of the marker is like reporting a crude association driven entirely by the
confounder. (4) Most convincing: geographic external validation on a centre with different
practices, then prospective evaluation, then subgroup analysis by scanner/protocol, then temporal
validation, then attribution maps (cheapest, weakest, and only works for localised shortcuts).
(5) A global shortcut: overall exposure, image noise level, or a reconstruction-kernel signature -
these are spread over the whole image and produce diffuse attribution maps that look
uninformative rather than suspicious.

**Reflection questions.** (1) For a manuscript: discrimination *and* calibration, with external
validation; attribution maps are supporting material at best. Before deployment: calibration in the
target population, subgroup performance, prospective evaluation, and monitoring - not heat maps.
(2) A data experiment, not a visualisation: for example evaluate on images where positioning and
exposure are matched between classes, or train on one centre and test on several others, or
manipulate the nuisance (crop the borders, normalise exposure) and observe whether performance
survives. (3) Likely explanations other than "detects pneumonia": leakage between the split and the
patient (multiple images per patient), a confounded acquisition or annotation signal, and
optimistic evaluation (threshold or hyperparameter selection on the same data; prevalence far from
the clinical setting). (4) The analogy holds in structure but breaks in the remedies: you cannot
"adjust" a CNN for a confounder by adding a covariate, because the confounder is inside the input.
You must change the data (multi-site, matched design), the input (crop/normalise), or the objective
(invariance penalty) - all of which discard information. (5) Defensible ranking: an external test
set from a different vendor or multi-centre training data first (both directly attack
transportability), prospective evaluation next, attribution figures last.
(6) **Internal AUC 0.94 / external AUC 0.71 with changing Grad-CAM.** Investigate *before*
retraining: scanner/protocol and reconstruction kernels; prevalence and case-mix shift; acquisition
artefacts and annotations that differ by site; demographic composition; label definition and
grading practice; selection into the imaging pathway; calibration drift; and shortcut features that
were predictive internally but absent externally. Retraining on the same design will often
reproduce the same failure. Fix the measurement and transportability problem first.

## Notebook 2

**Part 1 think-before-running.** No. Reconstruction quality measures preserved *pixel variance*, and
clinically useful information is often a tiny fraction of it. A perfect reconstruction of the
dominant variation can coexist with total loss of the diagnostic detail.

**Part 3 interpretation.** Preserved: overall colour and staining intensity, cell size, coarse
nucleus/cytoplasm contrast, position. Lost: fine chromatin texture, granule detail, sharp membrane
boundaries, small inclusions - which is much of what a haematologist actually grades. Yes, subtle
disease-relevant detail can vanish while the image still looks fine: a feature covering 1% of the
pixels contributes about 1% of the MSE, so the objective barely notices it. Per-class error tracks
visual complexity and heterogeneity (large, textured cells worst; platelets best) more than class
frequency in our balanced subset.

**Part 5 interpretation.** (1) Yes, partly: several types form visible groups, with overlap between
morphologically similar ones. (2) It demonstrates that cell type is a *major axis of pixel
variance*, so a variance-driven objective retains it - not that the model "understands" cell types.
(3) It does not demonstrate sufficiency for diagnosis, does not give the axes meaning, does not
make distances clinically calibrated, and says nothing about replication in another laboratory.
(4) No: a 2D projection with two thirds of the variance can still hide the structure that matters,
and the *remaining* third is where subtle phenotypes and technical effects often live.

**Part 6 interpretation.** (1) The nonlinear encoder usually wins modestly at equal dimensionality;
report the number rather than the intuition. (2) PCA is easier to report and to reproduce exactly;
the autoencoder depends on seed, architecture and stopping point. (3) A latent dimension has no
eigen-image, but you can decode perturbations along it (latent traversal) - that is the closest
analogue. (4) Neither on its own: prefer the clustering that is stable across representations,
seeds, and subsamples, and that predicts something external. (5) PCA of pixels keeps global
intensity structure exactly and linearly; the autoencoder discards fine high-frequency detail but
captures nonlinear shape/colour combinations that no linear projection can express.

**Part 7 interpretation.** (1) Both: the same latent space separates cell types *and* separates the
simulated sites, and the site classifier's high AUC quantifies the second part. (2) Clustering may
recover sites rather than subtypes; a model trained at one centre degrades at another; and latent
features used as exposures or covariates carry an acquisition component, so associations can be
driven by protocol rather than biology. (3) If site is associated with the outcome, site becomes a
classical confounder: `site -> image features` and `site -> outcome`, so the crude
feature-outcome association is biased and harmonisation that ignores the outcome can remove signal.
(4) Per-site standardisation is the weakest (assumes identical case mix); site as a covariate keeps
the information but does not make the model transportable; image-level harmonisation before encoding
is preferable because it cannot use the labels; adversarial training can work but risks removing
biology entangled with site. (5) No: a non-drop could simply mean the downstream task does not use
the site-encoding directions - the information is still there for the next analysis.

**Part 8 interpretation.** See the notebook: smoothness is a property of the decoder, alpha is not
time, intermediate points have no ground truth, and the interpolation direction mixes biology with
technical variation. Legitimate uses: sanity-checking a learned space, visual communication,
carefully validated augmentation, and sensitivity analysis - not trajectory claims.

**Part 9 interpretation.** (1) The label-free 32-dimensional probe typically lands well below the
supervised ResNet-18 benchmark (~0.958 accuracy at 28x28 in MedMNIST v2). The gap comes from three
places: the representation was optimised for pixel reconstruction, not discrimination; it has only
32 dimensions; and the probe is linear. (2) With many labels, supervised training wins; with few,
a frozen representation plus a linear head usually wins; with none, the representation is all you
have. (3) Confusions concentrate among morphologically similar types (immature granulocytes vs
neutrophils vs monocytes) - which is reassuring, whereas confusions structured by colour or size
alone would suggest a technical driver. (4) Adding reconstruction error might help slightly: it
encodes "unusualness", which correlates with rare and complex classes - but it is a quality
statistic, not a biological feature, so it will also import acquisition effects.

**Reflection questions.** (1) Before believing new subtypes: (i) show that the clusters are not
predicted by site/batch/scanner (repeat our site-prediction test), (ii) show stability across seeds,
subsamples, representations and cluster counts, (iii) show association with an *external* variable
not used in the representation (outcome, treatment response, independent assay). (2) What is
speaking is the dominant variance of the measurement process, which includes biology *and*
acquisition; "the data" does not distinguish them. (3) Prefer PCA when the sample is small,
when exact reproducibility and reporting matter, when the downstream analysis needs interpretable
loadings, or when a linear structure is adequate - for example a covariate-adjustment step in a
regression-based analysis. (4) Tell them the latent features are not identifiable and cannot be
mapped one-to-one onto biological factors: instead do latent traversals to see what a direction
changes visually, test whether the prediction survives removal of candidate nuisances, and then
design the follow-up around a measurable, pre-specified feature rather than around `z17`.
(5) For multicentre epidemiological use, the primary metric should be **performance under external
validation, reported alongside site-predictability of the representation**; reconstruction error and
silhouette are internal diagnostics that can look excellent while the representation is unusable
across centres.

---

# 10. Adaptations

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



