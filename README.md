# StarRegGAE
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/c3fb3c42-fa61-44d1-ac82-d24e13e50008" />
# StellarRegGAE

**Stellar Parameter Regression with MLP‑ResNet (Multitask)**

A clean, reproducible pipeline for multitask regression of stellar atmospheric parameters (Teff, [Fe/H], log g) from medium‑resolution optical spectra. It uses a compact MLP‑ResNet backbone with task‑specific heads, strong preprocessing/augmentation, and Bayesian hyperparameter tuning.

> Notebook: `StellarRegGAE.ipynb` (converted into a repo layout below).

## 🚀 Key Features

* **End‑to‑end pipeline**: loading → cleaning → normalization → augmentation → training → evaluation → persistence.
* **Multitask head**: predicts **Teff (K)**, **[Fe/H] (dex)**, and **log g (dex)** jointly from the same spectrum.
* **Compact MLP‑ResNet**: stacked Dense blocks with residual/skip connections, BatchNorm, Dropout.
* **Bayesian hyper‑tuning**: search over widths/depths/dropout/regularization/optimizer params with Keras Tuner.
* **Robust preprocessing**: IQR‑based filters, continuum‑normalized fluxes, label scaling, duplicate handling.
* **Data augmentation**: Gaussian noise, small wavelength shifts, low‑frequency “ripple” continuum perturbations.
* **Transparent metrics**: MAE and R² in **physical units** via inverse‑transformed scalers.
* **Reproducibility**: seeds set; models and scalers saved to disk.

## 🗂️ Repo Structure (suggested)

```
.
├── notebooks/
│   └── StellarRegGAE.ipynb
├── datasets/                 # place the .npy arrays here (not versioned)
├── models/                   # trained weights / artifacts
├── results/                  # figures, diagnostics
├── README.md
└── requirements.txt
```

## 📦 Data

The experiments expect pre‑generated NumPy tensors in `datasets/`:

* `X_raw_resampled_filtered_train.npy` — continuum‑normalized flux arrays (train)
* `y_raw_loaded_filtered_train.npy` — labels `[Teff, [Fe/H], log g]` (train)
* `X_raw_resampled_filtered_test.npy` — flux arrays (test)
* `y_raw_loaded_filtered_test.npy` — labels (test)

> Each **row** is a spectrum; each **column** is a wavelength bin.

If your file names differ, edit the respective paths in the notebook cells.

## 🧰 Requirements

Create a `requirements.txt` with:

```
joblib
keras
keras-tuner
matplotlib
numpy
scipy
scikit-learn
tensorflow>=2.12
```

Python ≥ 3.9 is recommended. TensorFlow should target your local CUDA/cuDNN stack if using GPU.

## ⚙️ Quickstart

1. **Clone & install**

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo>
   python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare data**

   ```
   mkdir -p datasets models results
   # copy your *.npy files into datasets/
   ```

3. **Run the notebook**

   ```bash
   jupyter lab  # or: jupyter notebook
   # open notebooks/StellarRegGAE.ipynb and run all cells
   ```

## 🧪 Model & Training Overview

* **Backbone**: MLP‑ResNet (Dense → BN → ReLU → Dropout) × N with skip connections.
* **Heads**: three lightweight Dense branches for Teff, [Fe/H], log g.
* **Loss**: per‑task **MSE** (sum across heads); report **MAE** and **R²**.
* **Tuning**: Keras Tuner (BayesianOptimization) over widths, depth, dropout, L2, learning rate, etc.
* **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best by val loss).
* **Saved artifacts**: model (`models/stellar_regression_model.keras`), scalers (`joblib`), metrics and plots to `results/`.

## 📈 Results & Diagnostics

The notebook includes:

* Distribution plots for labels and quality filters.
* Train/validation curves and per‑task MAE histories.
* Predicted vs. true scatter with 1:1 reference.
* Error histograms/densities (kde) and summary tables.

If you want to export images to the repo, save figures into `results/` (git‑tracked).

## 🔁 Reproducibility

* Fixed seeds via `random`, `numpy`, and TensorFlow.
* Label scalers persisted with `joblib`.
* Strict separation of train/test splits with optional duplicate checks at the **label** level.

## 🧩 Extending

* Swap in different augmentations (telluric masks, stronger continuum ripple).
* Add physics‑inspired priors or wavelength‑attention masks per parameter.
* Try Conv1D or spectral transformers for local pattern sensitivity.
* Convert to TF SavedModel / ONNX for deployment.

## 📜 Citation

If you use this repository in academic work, please cite the survey/data releases you rely on (e.g., SDSS/BOSS) and acknowledge Keras/TensorFlow, scikit‑learn, and SciPy.

## 📄 License

MIT — feel free to adapt to your needs. If you require a different license, update this section.

---

**FAQ**

**Q: Where do I put my spectra if I don’t have the provided `.npy` files?**
A: Add a small preprocessing script that reads FITS → resamples to the notebook’s grid → normalizes → saves `.npy` arrays in `datasets/`. Wire the paths in the first cells.

**Q: Can I run without GPU?**
A: Yes. The network is compact; CPU training will be slower but feasible for small experiments.

**Q: How do I reproduce the exact hyper‑parameters?**
A: The notebook seeds and search spaces are fixed; the best config is saved via `ModelCheckpoint` and can be reloaded from `models/`.


<img width="2126" height="592" alt="image" src="https://github.com/user-attachments/assets/f76a4850-1f4e-48e5-97bf-7b813f12134c" />
