# Instructor fallback weights

Committed checkpoints used when notebook `TRAIN_*` flags are `False` (the class default).

| File | Model | Notebook |
| --- | --- | --- |
| `cnn_pneumoniamnist.pt` | SmallCNN trained on clean PneumoniaMNIST subset | Practical 1 |
| `cnn_shortcut.pt` | SmallCNN trained on simulated marker-contaminated data | Practical 1 |
| `autoencoder_bloodmnist.pt` | ConvAutoencoder (`latent_dim=32`) on BloodMNIST subset | Practical 2 |

Training recipes match the notebooks (`SEED=0`, same subset sizes and hyperparameters).
Students do not need to regenerate these files; set `TRAIN_*=True` in a notebook if you want to retrain live.
