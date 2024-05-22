# Socially-Equitable-Public-Models
[ICML 2024] Building Socially-Equitable Public Models

This is the PyTorch implementation of our paper:

__Building Socially-Equitable Public Models__<br>
Yejia Liu, Jianyi Yang, Pengfei Li, Tongxin Li, Shaolei Ren<br>
The Forty-first International Conference on Machine Learning (ICML 2024)


[[arXiv](https://github.com/Ren-Research/Socially-Equitable-Public-Models)][[GitHub](https://github.com/Ren-Research/Socially-Equitable-Public-Models)]

## Installation
1. Create and activate a local conda environment:
```
conda create --name <env_name> python=3.7
conda activate <env_name>
```
3. Install PyTorch following the instructions on https://pytorch.org/ (we used PyTorch 1.5.1 in our experiments). For example,
```
conda install pytorch==1.5.1 torchvision==0.6.1 -c pytorch
```
4. Clone the repository and enter its root directory:  
```
git clone https://github.com/Ren-Research/Socially-Equitable-Public-Models.git
cd Socially-Equitable-Public-Models
```
5. Install required packages and dependencies in `requirement.txt`:
```
conda install --file requirements.txt
```
6. Download data following [this guide](https://github.com/Ren-Research/Socially-Equitable-Public-Models/blob/main/data/README.md)

## Training and Evaluation

### Run Trained Public Models for Paper Results
For reproducibility, we release our trained public models [here](https://github.com/Ren-Research/Socially-Equitable-Public-Models/tree/main/trained_public_models).
<br>
To run these trained models, please use the script [run_public_models.sh](https://github.com/Ren-Research/Socially-Equitable-Public-Models/blob/main/run_public_models.sh):
```
sh run_public_models.sh
```

### Train Your Own Public Models
* For the Data Center Workload application, here are example training commands:
```
# Train an equitable model with different lambda, similar groups
python main_dc_workload.py --training --lr 0.05 --n_epochs 50 --batch_size 128 --diff_lambda --q_idx 1.5

# Train an equitable model with same lambda, similar groups
python main_dc_workload.py --training --lr 0.05 --n_epochs 50 --batch_size 128 --q_idx 1.1

# Train a baseline model with different lambda, different groups
python main_dc_workload.py --training --lr 0.05 --n_epochs 100 --batch_size 128 --diff_lambda --baseline --diff_group_dist
```
* For the EV Charging application, here are example training commands:
```
# Train an equitable model with similar groups
python -m pdb main_ev_charging.py --training --lr 1e-4 --n_epochs 50 --batch_size 128 --q_idx 30

# Train a baseline model with different groups
python -m pdb main_ev_charging.py --training --lr 1e-4 --n_epochs 100 --batch_size 128 --diff_group_dist
```
### Visualization
We provide the code for our paper plots in the following Jupyter python notebooks:

* [dc_plots](https://github.com/Ren-Research/Socially-Equitable-Public-Models/blob/main/results/dc/plots.ipynb)
* [ev_plots](https://github.com/Ren-Research/Socially-Equitable-Public-Models/blob/main/results/ev/plots.ipynb)


## Citation
```
@article{SociallyEquitablePUblicModel_ICML_2024,
  title = {Building Socially-Equitable Public Models},
  author = {Liu, Yejia and Yang, Jianyi and Li, Pengfei and Li, Tongxin and Ren, Shaolei},
  journal = {ICML},
  year = {2024}
}
```

