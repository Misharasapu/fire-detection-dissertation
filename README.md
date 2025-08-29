# Fire Detection Dissertation

This repository is to support the MSc project **"Optimising Synthetic Data Use in AI for Industrial Fire Detection"**, carried out as part of the MSc Data Science programme at the University of Bristol in collaboration with SYNOPTIX Ltd.

The project investigates whether synthetic imagery can reduce reliance on scarce real data while sustaining deployment-grade performance for industrial fire detection. The task is formulated as binary image classification (fire vs. no fire) using a ResNet-50 backbone, with experiments organised into four phases: frozen training, fine-tuning, domain shift evaluation, and indoor deployment.

---

## Relation Between Files

- **Notebooks**  
  Jupyter notebooks contain the main experiment workflows. They are designed to be run in order:
  - `01_setup_environment.ipynb` — initial environment setup (dependencies, paths, repo cloning).  
  - `02_data_preprocessing.ipynb` — preprocessing and label harmonisation for real and synthetic datasets.  
  - `03_train_resnet_real_only.ipynb` — training with real datasets only (D-Fire or PLOS ONE).  
  - `04_train_resnet_synthetic_only.ipynb` — training with synthetic datasets only (Yunnan or SYN-FIRE).  
  - `05_train_resnet_mixed_ratios.ipynb` — experiments with mixed synthetic–real ratios (25%, 50%, 75%).  
  - `06_train_resnet_finetuned.ipynb` — fine-tuned outdoor models, unfreezing `layer4` and the classification head.  
  - `07_train_resnet_indoor_models.ipynb` — indoor deployment experiments (PLOS ONE real-only, and PLOS ONE + SYN-FIRE 50/50 mix).  
  - `evaluate_models.ipynb` — evaluation pipeline, including metrics, confusion matrices, and Grad-CAM visualisations.  


- **Utils**  
  Python helper modules providing reusable functionality:
  - `fire_classification_dataset.py` — dataset loaders for real, synthetic, and mixed training sets.  
  - `train_model.py` — modular training loop supporting frozen and fine-tuned regimes.  
  - `metrics.py` — metrics for evaluation, including accuracy, precision, recall, F1, MCC, ROC AUC, and PR AUC.  

---

## Data

The experiments use both real and synthetic fire datasets (not included in this repository due to size and licensing restrictions):

- **D-Fire** (outdoor, real) — [Venâncio et al., 2022, *Sensors*]  
- **Indoor Fire and Smoke (PLOS ONE)** (indoor, real) — [Sozol et al., 2025, *PLOS ONE*]  
- **Yunnan MSFFD** (outdoor, synthetic) — [Hu et al., 2023, *CVPR Workshops*]  
- **SYN-FIRE** (indoor, synthetic) — [Arlovic et al., 2025, *Fire Technology*]  

All datasets were harmonised into a binary classification format (fire vs. no fire) for experiments. Full details are described in the dissertation.

---

## Requirements

A `requirements.txt` file should be included to set up the environment. Core libraries include:

- Python 3.10  
- PyTorch + torchvision  
- numpy, pandas, matplotlib  
- scikit-learn  

---

## Thesis Linkage

This repository is directly referenced in the MSc dissertation. The dissertation links to:
- This GitHub repository (for code and notebooks).  
- Public sources of the datasets used in the experiments.  

---

✍️ **Author**: Mishara Sapukotanage  
📖 MSc Data Science, University of Bristol (2025)  
