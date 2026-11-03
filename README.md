# Adaptive Anti-Class CIL (ICECS 2026)

Official PyTorch implementation of **"Continual Learning by Exploiting Adaptive Anti-Class Distributions"** (IEEE ICECS 2026).

**Authors:** Christos Tsingiropoulos, Dimitrios Katsikas, Nikolaos Passalis, Anastasios Tefas — Aristotle University of Thessaloniki.

> In Class-Incremental Learning (Class-IL), as new classes arrive, the features lose the well-separated structure of Neural Collapse, so old and new classes start to overlap and the model forgets. To keep this structure, this work brings anti-class supervision into Class-IL with a confusion-guided version of the One-Cold Cross-Entropy (OCCE) loss, which pushes each class away from the classes the model confuses it with the most. The loss is added to [DER](https://arxiv.org/abs/2103.16788) without adding any network, module or trainable parameter.

## Method

For a sample with label `y`, when `C` classes have been seen so far, the training objective is:

```
loss      = CE + L_aux + gamma · loss_anti
loss_anti = - Σ_k Tmix[k] · log softmax(-logits)[k]
Tmix      = (1 - α) · U + α · prior[y]
```

- `softmax(-logits)` gives the anti-class probabilities, which are high for the classes that the sample does not belong to.
- `Tmix` is the anti-class target of the sample. `U` is the uniform target, which is zero for class `y` and `1/(C-1)` for every other class. `prior[y]` is the row of class `y` in the confusion matrix, with the diagonal set to zero and the row normalized again, so it shows which classes the model predicts when it makes a mistake on class `y`. The confusion matrix is computed again at the end of every epoch.
- The larger `Tmix[k]` is, the further class `k` is pushed away from class `y`, so the classes that the model confuses the most with `y` are pushed apart more strongly. With `α = 0` (`prior_a`) the loss is the uniform OCCE loss, and with `gamma = 0` it is plain DER.

The loss is in [`utils/occe_probabilities.py`](utils/occe_probabilities.py) (`OnlineOCCELoss`) and it is added to DER in [`models/der.py`](models/der.py). This version of DER does not include the balanced classifier re-training stage of the original DER. Instead, as in PyCIL, the weights of the classifier are aligned at the end of every incremental task.

## Results

Last and Average Accuracy (%). The results of DER and TagFex are taken from their papers, and DER uses a simpler data augmentation. The proposed method uses the same AutoAugment policy and training schedule as TagFex, but with one augmented view of every image instead of two.

| Method | CIFAR-100 B10-I10 | CIFAR-100 B50-I10 | ImageNet-100 B10-I10 | ImageNet-100 B50-I10 | ImageNet-1000 B100-I100 |
|:---|:---:|:---:|:---:|:---:|:---:|
| | Last / Avg | Last / Avg | Last / Avg | Last / Avg | Last / Avg |
| DER | 64.35 / 75.36 | 65.27 / 72.60 | 66.71 / 77.18 | 71.08 / 77.71 | 58.83 / 66.87 |
| TagFex | 68.23 / **78.45** | 70.33 / 75.87 | 70.84 / 79.27 | 75.54 / 80.64 | 61.45 / 68.32 |
| Proposed | **69.35** / 77.54 | **72.06** / **77.17** | **71.95** / **79.74** | **77.55** / **82.27** | **61.83** / **70.10** |

## Setup

```bash
pip install -r requirements.txt
```

The code was developed with Python 3.8 and PyTorch 1.12.

### Data

- **CIFAR-100** is downloaded automatically to `./data`.
- **ImageNet-100** is the standard ImageNet subset used by UCIR, PODNet and PyCIL. The list of classes and files can be found [here](https://drive.google.com/drive/folders/1RBrPGrZzd1bHU5YG8PjdfwpHANZR_lhJ?usp=sharing) (from PyCIL). It should be placed in `ImageFolder` format at `./data/imagenet100/train` and `./data/imagenet100/val`, with only these 100 classes.
- **ImageNet-1000** is the full ImageNet, placed at `./data/imagenet/train` and `./data/imagenet/val`.

If your data is somewhere else, change the paths in [`utils/data.py`](utils/data.py).

All datasets are trained with AutoAugment, the CIFAR-10 policy for CIFAR-100 and the ImageNet policy for ImageNet-100 and ImageNet-1000.

## Running

```bash
# Proposed method, DER with the confusion-guided anti-class loss
python main.py --config=./exps/der_anti_class_cifar100_b10_i10.json
python main.py --config=./exps/der_anti_class_cifar100_b50_i10.json
python main.py --config=./exps/der_anti_class_imagenet100_b10_i10.json
python main.py --config=./exps/der_anti_class_imagenet100_b50_i10.json
python main.py --config=./exps/der_anti_class_imagenet1000_b100_i100.json

# DER in the same code base, without the anti-class loss
python main.py --config=./exps/der_cifar100_b10_i10.json
```

On CIFAR-100 and ImageNet-100, every experiment is run with the seeds 0, 1 and 2, and the reported results are the mean of the three runs. These seeds give the fixed class orders of the paper (seed 1 is the iCaRL order), while any other seed gives a random order. On ImageNet-1000, the class order of DER is used (seed 1993). The logs are saved in `logs/` and the checkpoints of every task in `checkpoints/`.

### Hyperparameters

| Dataset | `prior_a` (α) | `gamma` (γ) |
|:---|:---:|:---:|
| CIFAR-100 | 0.2 | 0.5 |
| ImageNet-100 | 0.5 | 0.5 |
| ImageNet-1000 | 0.5 | 0.5 |

The same values are used for both protocols of each dataset. The method is not sensitive to γ, and α works best at a moderate value. On ImageNet-1000, uniform targets (α = 0) give a higher Last Accuracy (62.75 vs. 61.83), because with 100 new classes per step the confusion matrix is very noisy. Smaller values of α (e.g. 0.1–0.3) were not tested on ImageNet.

### Main options

All the options are set in the JSON configuration file that is given to `main.py` with `--config`. The files used in the paper are in [`exps/`](exps/), and the main keys are the following:

| Key | Description |
|---|---|
| `occe_criterion` | `confusion_anti_class` for the proposed method, `uniform` for OCCE with uniform anti-class targets, or `none` for plain DER |
| `gamma` | List of weights for the anti-class loss, with one run for each value |
| `prior_a` | α, the part of the anti-class target that comes from the confusion prior |
| `seed` | List of seeds, with one run for each seed |
| `init_cls`, `increment` | Number of classes in the first task and in each of the next tasks |
| `convnet_type` | `resnet18` (default), `resnet32`, `resnet34` or `resnet50` |
| `device` | List of GPU ids, e.g. `["0", "1"]`, or `[-1]` for CPU |
| `load_from_task` | Continue a run from the checkpoint saved after that task |

<!-- ## Future work
Two possible extensions of this work, which still change only the training objective and not the architecture, are the following.
- A dynamic α, which becomes smaller in the later steps, when the confusion matrix is sparser and less reliable.
- A different α for each class, based on how many of its samples are predicted correctly. In this way, the anti-class target would take into account not only the errors of the model, but also its correct predictions. -->

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tsingiropoulos2026icecs,
    author    = {Tsingiropoulos, Christos and Katsikas, Dimitrios and Passalis, Nikolaos and Tefas, Anastasios},
    title     = {Continual Learning by Exploiting Adaptive Anti-Class Distributions},
    booktitle = {Proc. IEEE 33rd International Conference on Electronics, Circuits and Systems (ICECS)},
    year      = {2026}
}
```

This code is also part of my undergraduate thesis at the Department of Informatics, Aristotle University of Thessaloniki.

## Acknowledgments

This code is built on [PyCIL](https://github.com/G-U-N/PyCIL) (MIT License). The data pipeline, the DER implementation, the backbones and the exemplar memory come from PyCIL, so please also cite:

```bibtex
@article{zhou2023pycil,
    author  = {Da-Wei Zhou and Fu-Yun Wang and Han-Jia Ye and De-Chuan Zhan},
    title   = {PyCIL: a Python toolbox for class-incremental learning},
    journal = {SCIENCE CHINA Information Sciences},
    year    = {2023},
    volume  = {66},
    number  = {9},
    pages   = {197101},
    doi     = {https://doi.org/10.1007/s11432-022-3600-y}
}
```
