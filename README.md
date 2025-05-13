# Reaction-Diffusion Based Graph Neural Models for Traffic Forecasting

This repository contains source code and datasets for a series of models built upon reaction-diffusion graph dynamics for traffic forecasting:

- **RDGCN**: Reaction-Diffusion Graph Convolutional Network  
- **RDGODE**: Reaction-Diffusion Graph Neural ODE Network  
- **DRDGODE**: Dynamical Reaction-Diffusion Graph Neural ODE Network  
- **GreyRDGODE**: Grey-box Reaction-Diffusion GODE (experimental)

## Datasets

This project uses three benchmark traffic datasets:

- **METR-LA** and **PEMS-BAY** from [DCRNN](https://arxiv.org/abs/1707.01926)
- **Seattle Loop Detector** dataset: [GitHub link](https://github.com/zhiyongc/Seattle-Loop-Data)
- More data: [PEMS California DOT](https://pems.dot.ca.gov/)

Download all datasets from:  
📁 [Google Drive Folder](https://drive.google.com/drive/folders/1mZ5eRRS95lvFuYD6ZgIOIVc-2rirxUsG?usp=drive_link)

---

## 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Neural ODE solver (`torchdiffeq`):

```bash
pip install git+https://github.com/rtqichen/torchdiffeq.git
```

---

## 2. Train the Model (Weekday Data)

To train the model using weekday data:

```bash
bash run_seq.sh
```

You can edit `run_seq.sh` to run specific models. Example command:

```bash
python train_with_weekday_seq.py \
  --data=metrlaweekdayweekend \
  --device=cuda:0 \
  --predicting_point=-1 \
  --filter=0 \
  --learning_rate=0.005 \
  --weight_decay=0.001 \
  --epochs=2000 \
  --enable_bias=True \
  --num_sequence=13 \
  --num_weekday=36 \
  --start_runs=0 \
  --runs=6 \
  --num_rd_kernels=2 \
  > logs/train.log 2>&1 &
```

### Parameter Descriptions:

- `data`: Dataset name (`metrlaweekdayweekend`, `pemsbayweekdayweekend`, `seattleweekdayweekend`)
- `device`: GPU device id
- `predicting_point`:  
  - `-1`: multi-step prediction (default for RDGODE/DRDGODE)  
  - `1`: single-step prediction (for RDGCN)
- `filter`: If `0`, training uses all values including zeros
- `learning_rate`, `weight_decay`, `epochs`: standard training hyperparameters
- `enable_bias`: Whether to include bias in RD equation
- `num_sequence`: Length of output sequence (usually `13`, use last 12 for evaluation)
- `num_weekday`: Number of weekday samples for training
- `runs`: Number of model training runs (for different splits)
- `num_rd_kernels`:  
  - `1` → RDGODE  
  - `>1` → DRDGODE

---

## 3. Evaluate on Weekend Data

Use `test_with_weekend_seq.py` with the same parameters used for training:

```bash
python test_with_weekend_seq.py --data=metrlaweekdayweekend --device=cuda:0 ...
```

**Note**: The saved model name is hardcoded in `train_with_weekday_seq.py`, line 118. Modify as needed.

---

## Notes

- The GreyRDGODE model is included but lacks a complete interface.
- Evaluation results will be printed during testing.
