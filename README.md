# ProbabilisticActionSegmentRefiener
The following repo contains code for Action Segmentation Regulator that uses Probabilistic models to detect erroneous action segmentstions and OOD data instances

We followed the [Diffusion Action Segmentation](https://arxiv.org/abs/2303.17959) (ICCV 2023) paper and used their code at [DiffAct](https://finspire13.github.io/DiffAct-Project-Page/) as a base for our work.

## Environment
Python 3.9+
Pytorch Latest version CUDA 11.3+
networkx
numpy
scipy
matplotlib
tqdm
json

## Data

* Download features of 50salads, GTEA and Breakfast provided by [MS-TCN]() and [ASFormer](https://github.com/ChinaYi/ASFormer): [[Link1]](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) [[Link2]](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)
* Unzip the data, rename it to "datasets" and put into the current directory
```
ProbabilisticActionSegmentRefiner/
├── datasets
│   ├── 50salads
│   │   ├── features
│   │   ├── groundTruth
│   │   ├── mapping.txt
│   │   └── splits
│   ├── breakfast
│   │   ├── features
│   │   ├── groundTruth
│   │   ├── mapping.txt
│   │   └── splits
│   └── gtea
│       ├── features
│       ├── groundTruth
│       ├── mapping.txt
│       └── splits
├── main.py
├── model.py
└── ...
```

## Run the masked inferences

* Generate config files by `python default_configs.py`
* Run `python main.py --config configs/some_config.json --device gpu_id --is_train 0`
* If you want to train the model again, set `--is_train 1`
* Trained models and logs will be saved in the `result` folder

## OOD experiment
* Run `python ablation/transitiondurationAbalation.py` for GTEA dataset
* Run `python ablation/transitiondurationBreakfastAblation.py` for Breakfast dataset
* Run `python ablation/transitiondurationSaladAblation.py` for 50salads dataset

## Model Training
* To train a model from scratch, please use the corresponding JSON for your dataset and split. Here, we give an example of training on GTEA Split 1. In the end, we use the best model for our inference.
```
python main.py --config configs/GTEA-Trained-S1.json --is_train 1
```
* We also provide some trained models in the `trained_models` folder to use for inference

## Cross Domain Inference
* Please use the --config and --model_path from the source domain and --dataset_name from the target domain. The following code will save the result to the folder result1/50salads-Trained-S1-Tested-GTEA-S1.
```
python main1.py --config configs/50salads-Trained-S1.json --result_dir result1 --sample_rate 1 --action test --naming 50salads-Trained-S1-Tested-GTEA-S1 --model_path trained_models/50salads-Trained-S1/release.model --dataset_name gtea
```

## Visualization Fig.5
* By default, it takes results in the rafid/GTEA-Trained-S1/ and saves the plot in the rafid/GTEA-Trained-S1/plot. You can change to a different dataset or split.
```
python rafid_v1.py --split 1
```
