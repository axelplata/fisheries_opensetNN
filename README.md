# Continuous Learning

**Novelty Recognition: Fish Species Classification via Open-Set Recognition**

Code underlying the paper:

> Córdova, M.; Torres, R.d.S.; van Helmond, A.; Kootstra, G.
> *Novelty Recognition: Fish Species Classification via Open-Set Recognition.*
> **Sensors** 2025, 25, 1570. https://doi.org/10.3390/s25051570

## Description

In automated fish-catch registration, computer-vision systems are usually trained as
**closed-set** classifiers: they can only recognise the species seen during training. In
the real world, fish from species that were never in the training data appear in new
regions and seasons, and a closed-set model is forced to (mis)label them as one of the
known species — corrupting the per-species counts and discarding new, useful samples for retraining.

This repository treats fish classification as an **open-set recognition** problem: the
classifier may assign a sample either to one of the *known* species **or** to an
*"unknown"* class. It compares three open-set methods on the publicly available
**Fish Detection and Weight Estimation (FDWE)** dataset (1086 images, 2216 fish
instances, 9 North-Sea species):

| Method | Folder | Idea | Input |
| --- | --- | --- | --- |
| **OSNN** — Open-Set Nearest Neighbor | [`OSNN/`](OSNN/) | Rejects a sample as unknown when the *ratio* of its distance to the two nearest samples of **different** classes, `R = d(p,q)/d(p,s)`, exceeds a tuned threshold. | Pre-extracted ResNet-18 features |
| **PISVM / W-SVM** — Probability-of-Inclusion SVM | [`svm_openset/`](svm_openset/) | One-vs-all SVMs whose decision scores are calibrated with a Weibull (Extreme Value Theory, via libMR) "probability of inclusion"; samples below a probability threshold are rejected. | Pre-extracted ResNet-18 features |
| **MGPL** — Multiple Gaussian Prototype Learning | [`MGPL/`](MGPL/) | A VAE whose latent space is shaped by per-class Gaussian prototypes; a sample is rejected when its KL-divergence to the nearest prototype exceeds a tuned threshold. | Raw fish image crops (extracts its own features) |

### Results from the paper

*Closed-set* (4 known species only):

| Method | F1-macro |
| --- | --- |
| OSNN (KNN, K = 3) | **0.97** |
| PISVM | 0.90 |
| MGPL | 0.87 |

*Open-set* (4 known species + Unknown), mean ± std :

| Method | F1-macro (5-class) | Binary F1-macro (known vs unknown) | AUROC |
| --- | --- | --- | --- |
| **OSNN** | **0.79 ± 0.05** | **0.84 ± 0.06** | **0.92 ± 0.01** |
| PISVM | 0.73 ± 0.04 | 0.82 ± 0.05 | 0.89 ± 0.02 |
| MGPL | ≈ 0.45 | — | 0.74 ± 0.03 |

### Repository structure

```
fisheries_opensetNN/
├── features/                       # Shared input features (unzip before use)
│   └── resnet18.zip                #   → resnet18/{train,val,test}.npy (512-d ResNet-18/ImageNet features + labels)
├── OSNN/                           # Open-Set Nearest Neighbor (single Jupyter notebook)
│   └── Openset_fish_OSNNPaper2025.ipynb
├── svm_openset/                    # PISVM / W-SVM (open-set SVM: C/C++ engine + Python drivers)
│   ├── libsvm-openset/             #   native engine (build with `make`): svm-train / svm-predict / svm-scale
│   ├── diduch_code/                #   current Python pipeline + datasets/ + trained models/
│   └── W-SVM/                      #   earlier duplicate pipeline + libMR/ (Weibull/EVT) + README.md
├── MGPL/                           # Multiple Gaussian Prototype Learning (image-based, PyTorch)
│   ├── 2_README_OpenSetMGPL.txt    #   method-specific usage notes from the author
│   ├── train.py / test_*.py / find_threshold_*.py
│   ├── data/ networks/ save_model/ figure/MGPL.jpg
│   └── ...
├── 1_partitions.txt                # Species index map + the 5 open-set partitions
├── LICENSE                         # CC BY-SA 4.0
├── en-co-fundedvertical-rgb-pos.png
└── Readme                          
```

### Dataset

The **FDWE** dataset is **not** bundled here. It is publicly available at
**DOI [10.4121/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3](https://dx.doi.org/10.4121/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3)**.
For OSNN and the SVM methods you only need the pre-extracted features shipped in
[`features/resnet18.zip`](features/); MGPL needs the raw image crops (see below).

**Citation** — if you use this code, please cite the paper:

```bibtex
@article{cordova2025novelty,
  title   = {Novelty Recognition: Fish Species Classification via Open-Set Recognition},
  author  = {C{\'o}rdova, Manuel and Torres, Ricardo da Silva and van Helmond, Aloysius and Kootstra, Gert},
  journal = {Sensors},
  volume  = {25},
  number  = {5},
  pages   = {1570},
  year    = {2025},
  doi     = {10.3390/s25051570}
}
```

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)

## Installation

> NOTE: This is **research code**. There is no single installer and no pinned
> `requirements.txt`; each method has its own environment and several scripts contain
> hard-coded paths which must be changed (see the per-method caveats under [Usage](#usage)). The
> original reference environment was the `ultralytics/ultralytics:latest` Docker image
> (as of January 2025).

**1. Clone the repository 

```bash
git lfs install
git clone https://github.com/axelplata/fisheries_opensetNN.git
cd fisheries_opensetNN
```

**2. Unpack the shared features** (needed by OSNN and the SVM methods):

```bash
unzip features/resnet18.zip -d features/
# → features/resnet18/{train,val,test}.npy
```

Each `.npy` is a 2-D array whose columns are `512 ResNet-18 (ImageNet) features` +
`species_index` + `image_name`. The arrays are stored as **strings**.

**3. Python environment** (packages inferred from the source; versions are not pinned):

```bash
conda create -n openset python=3.10
conda activate openset
pip install numpy scipy scikit-learn pandas matplotlib seaborn pillow tqdm jupyter \
            torch torchvision hyperopt
```

- `torch` / `torchvision` and a **CUDA-capable GPU** are required for **MGPL**.
- `jupyter` is needed for **OSNN**.
- `hyperopt` is used by the **SVM** threshold/hyperparameter search.

**4. Build the native SVM engine**:

```bash
# requires: g++, make, cmake (>2.8), build-essential
cd svm_openset/libsvm-openset
# edit line 2 of the Makefile: set LIBMR_DIR to the ABSOLUTE path of the libMR folder
#   (e.g. <repo>/svm_openset/W-SVM/libMR)
make            # builds libMR (via cmake) + svm-train, svm-predict, svm-scale
```

## Usage

All experiments use the same **open-set partitioning**, defined in
[`1_partitions.txt`](1_partitions.txt). Of the nine FDWE species, the **four majority
species are always the *known* classes** and the remaining five are held out as *unknown*
(two used to tune thresholds on the validation set, three to test). Species are referred
to by their FDWE index:

| FDWE idx | Species | Role |
| --- | --- | --- |
| 0 | *Pleuronectes platessa* | known |
| 1 | *Solea solea* | known |
| 3 | *Eutrigla gurnardus* | known |
| 5 | *Limanda limanda* | known |
| 2 | *Scophthalmus rhombus* | unknown |
| 4 | *Scophthalmus maximus* | unknown |
| 6 | *Amblyraja radiata* | unknown |
| 7 | *Merlangius merlangus* | unknown |
| 8 | *Scyliorhinus canicula* | unknown |

The **5 experiments** all keep `known = [0, 1, 3, 5]` and vary the unknowns:

| # | `unknown_val` | `unknown_test` |
| --- | --- | --- |
| 1 | 2, 4 | 6, 7, 8 |
| 2 | 6, 8 | 2, 4, 7 |
| 3 | 2, 7 | 4, 6, 8 |
| 4 | 4, 8 | 2, 6, 7 |
| 5 | 6, 7 | 2, 4, 8 |


---

### OSNN — `OSNN/`

A self-contained Jupyter notebook,
[`OSNN/Openset_fish_OSNNPaper2025.ipynb`](OSNN/Openset_fish_OSNNPaper2025.ipynb), built on
the pre-extracted ResNet-18 features.

1. Make the features available at `data/features/resnet18/{train,val,test}.npy`. The first
   notebook cells fetch them from an **internal WUR Git** (`git clone … data` +
   `unzip`); simply reuse the bundled
   [`features/resnet18.zip`](features/) — unzip it and point `features_dir` at it.
2. Open and run the notebook:
   ```bash
   cd OSNN
   jupyter notebook Openset_fish_OSNNPaper2025.ipynb   # or: jupyter lab
   ```
3. Set `split_used` (0–4) to choose the experiment, then **Run All**. The notebook:
   - builds a **closed-set** baseline with KNN (`K = 3`, `metric='precomputed'`);
   - tunes the OSNN distance **ratio threshold** on validation by grid search
     (`np.arange(0.1, 1.0, 0.001)`), maximising the binary known-vs-unknown F1-macro
     (e.g. 0.626 in the saved run);
   - runs OSNN inference on validation and test and reports accuracy, F1-macro,
     binary F1-macro, **AUROC**, and confusion-matrix PDFs.

**Notes / caveats**
- `split_used` is hard-coded; to reproduce all five experiments, loop it over `range(5)`
  (or change it and re-run) and aggregate the printed metrics.
- The `!git clone` / `!unzip` cells assume a Unix-like shell; on Windows, prepare
  `data/features/resnet18/` manually.

---

### PISVM / W-SVM — `svm_openset/`

Open-set SVMs built on a modified **libsvm-openset** linked
against **libMR** for Weibull/EVT calibration. The same native binary exposes several
algorithms via `svm-train -s`: **W-SVM** (`-s 8`), **PI-SVM** (`-s 10`), and **1-vs-set**
(`-s 7`).

> **Canonical path:** the current pipeline is **`svm_openset/diduch_code/code/`** driving
> the engine in **`svm_openset/libsvm-openset/`**, with data and models under
> `svm_openset/diduch_code/datasets/` and `…/models/`. The `svm_openset/W-SVM/` folder is
> an **older, self-contained duplicate**.

After building the engine ([Installation](#installation) step 4):

```bash
cd svm_openset/diduch_code/code

# 1. Convert feature CSVs → libsvm sparse format (per split; column 0 is the label)
python convert_csv_to_libsvm.py \
    --csv_file ../datasets/fdwe/csvs/split_0/train.csv \
    --output_dir ../datasets/fdwe/libsvm_format/split_0/
#   (repeat for val.csv and test.csv, and for split_1 … split_4)

# 2. Train a closed-set model on the known species (W-SVM shown; PI-SVM = *_pisvm.py)
python train_closed_set_wsvm.py            # → models/closed_set/wsvm/one_vs_rest_G<γ>_C<C> (+ _one_wsvm)

# 3. Tune the open-set probability threshold on the validation set
python optimize_open_set_wsvm.py --split 0 --gamma 0.01 --C 8

# 4. Evaluate: closed-set (known only) and open-set (with rejection) metrics
python predict_closed_set_wsvm.py
python predict_open_set_wsvm.py --split 0 --gamma 0.01 --C 8 --thres <best_p> --partition test
```

The per-split feature CSVs (`datasets/fdwe/csvs/split_{0..4}/{train,val,test}.csv`), their
libsvm conversions, and a few pre-trained models (W-SVM `G0.01_C8`, PI-SVM `G2.91_C4`,
1-vs-set `G4.86_C10`) are already included.

**Notes / caveats**
- The root notes state these methods were *"copied from the docker container, need to be
  checked"* — treat them as Docker-extracted and **not re-verified outside that container.**
- **Every script hard-codes absolute Docker paths** (`/usr/src/svm_openset/…`); you must
  either run inside the original image or edit the paths.
- The native build is **Linux/macOS** (needs `make`, `g++`, `cmake`). Set `LIBMR_DIR` to match the Makefile you
  build.
- The "unknown" label is **99** throughout; datasets must use 99 for unknown classes.
- `predict_*` scripts reference a model named after `--gamma`/`--C`, so pass values for
  which a trained model actually exists.

---

### MGPL — `MGPL/`

A conditional VAE with per-class Gaussian prototypes (adapted from
[LiuJMzzZ/MGPL](https://github.com/LiuJMzzZ/MGPL)). Unlike the other two methods it works
on **raw image crops** and extracts its own features. See
[`MGPL/2_README_OpenSetMGPL.txt`](MGPL/2_README_OpenSetMGPL.txt) for author's full notes.

1. **Stage the data** as an ImageFolder tree, with per-class subfolders named by integer
   class id:
   ```
   MGPL/data/fdwe/{train,val,test}/<class_id>/<image>.png
   ```
2. **Define the splits** you want in [`MGPL/data/splits.py`](MGPL/data/splits.py)
   (`known_classes`, `unknown_val`, `unknown_test`).
3. **Inject unknowns** into the val/test folders with `./prepare_data.sh` (edit the
   hard-coded `images_path` first). This copies *all* instances of the unknown species
   into the validation/test sets.
4. **Train** one model per split (GPU required):
   ```bash
   cd MGPL
   python train.py --dset fdwe --split 0 --h 512 --batch_size 16
   #   → save_model/fdwe_split_0.pt   (a pre-trained split-0 checkpoint is included)
   ```
5. **Choose a rejection threshold** on the validation set. Several strategies are provided;
   the paper used the AUROC-based threshold:
   ```bash
   python find_threshold_kl.py  --dset fdwe --split 0 --subset val \
       --model_dir save_model/fdwe_split_0.pt
   # alternatives: find_threshold_auc.py, find_threshold_bySpecie_kl.py,
   #               find_threshold_byPrototype_allUnknowns.py, find_ratio_threshold.py …
   ```
6. **Test** with the chosen threshold:
   ```bash
   python test_ours.py --dset fdwe --split 0 --subset test \
       --threshold <value_from_step_5> \
       --model_dir save_model/fdwe_split_0.pt --joinUnknowns
   #   → closed/open/binary/global metrics, AUROC, conf_matrix_*.pdf, AUC_ROC.pdf
   ```
   Optional diagnostics: `extract_features.py` + `tsne.py` (t-SNE of the latent space),
   `plot_reconstructed_img.py` (VAE reconstructions).

**Notes / caveats**
- Provenance of the exact paper version is not guaranteed.
- Scripts contain **hard-coded absolute paths** (e.g. `/data/manuel_…/…`) in the
  `*.sh` drivers and example `--model_dir` arguments — adjust for your environment.
- **A GPU is mandatory.** Checkpoints are saved as whole pickled objects
  (`torch.save(model)`), so loading needs the same `model.py`/`networks` on the path and a
  compatible torch version.
- `prepare_data.sh`/`copy_unknowns.py` mix train+val+test instances of unknown species
  into the val/test folders, so **restore the original folders between experiments**.
- Some testers use **hard-coded threshold literals** inside the `.py`
  (`test_bySpeciesThreshold_kl.py`, `test_byPrototypeThreshold_dist.py`) — paste in the
  values printed by the matching `find_threshold_*` script.

## License

This work is licensed under the **Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0)** license — see [`LICENSE`](LICENSE).

Bundled third-party components carry their own licenses (e.g. `svm_openset/libsvm-openset`
and `libMR` under `svm_openset/W-SVM/libMR/`); review those before redistribution.

## Acknowledgement

![Co-funded by the European Union](en-co-fundedvertical-rgb-pos.png)

This work was funded by the European Maritime and Fisheries Fund (contract number 16302)
under the Fully Documented Fisheries (FDF) project, and the European Union Horizon Europe
under the EVERYFISH project (grant agreement no. 101059892) and OptiFish project (grant
agreement no. 101136674).

## Contact

Axel Streit — axel.streit@wur.nl
