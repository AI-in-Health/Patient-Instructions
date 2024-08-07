# Patient Insturction Generation
<a href="https://zenodo.org/badge/latestdoi/544446958"><img src="https://zenodo.org/badge/544446958.svg" alt="DOI"></a>
![GitHub Repo stars](https://img.shields.io/github/stars/AI-in-Hospitals/Patient-Instructions)
![GitHub Repo forks](https://img.shields.io/github/forks/AI-in-Hospitals/Patient-Instructions)

Code for our paper published in NeurIPS 2022 [[arXiv]](https://arxiv.org/abs/2210.12777):
> **Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions**
> 
> (a.k.a., **Retrieval-Augmented and Knowledge-Grounded Language Models for Faithful Clinical Medicine**)
> 
> Fenglin Liu, Bang Yang, Chenyu You, Xian Wu, Shen Ge, Zhangdaihong Liu, Xu Sun*, Yang Yang*, and David A. Clifton.

## Updates
- `[22-10-25]`: We release the code and data.

## Clone the repo
```
git clone https://github.com/AI-in-Hospitals/Patient-Instructions.git

# clone the following repo to calculate automatic metrics
cd Patient-Instruction
git clone https://github.com/ruotianluo/coco-caption.git 
```

## Environment

```
conda create -n pi python==3.9
conda activate pi
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.10.0
pip install pytorch-lightning==1.5.1
pip install pandas rouge scipy

# if you want to re-produce our data preparation process
pip install scikit-learn plotly
```
Higher version of `torch` and `cuda` can also work.



## Download the data 
As we can not re-distribute the raw [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) data, we release only our pre-processed dataset used in the paper at [Google Drive](https://drive.google.com/file/d/1z1SvPDZ_yixuWuzQr9aK7bNPJUq2tEhY/view?usp=sharing) (`data.zip`, 132MB). After downloading, unzip the data and place it like the structure below:

```
Patient-Instructions/ # the root of the repo
    data
    ├── README.md
    ├── prepare_dataset.ipynb
    ├── prepare_subtasks.ipynb
    ├── diagnose-procedure-medication
    │   ├── admDxMap_mimic3.pk      # Source: D_ICD_DIAGNOSES.csv
    │   ├── admMedMap_mimic3.pk     # Source: prescriptions.csv
    │   ├── admPrMap_mimic3.pk      # Source: D_ICD_PROCEDURES.csv
    │   └── readme.txt
    ├── splits                      # Source: NOTEEVENTS.csv
    │   ├── train.csv               # obtained by data/prepare_dataset.ipynb
    │   ├── val.csv                 # obtained by data/prepare_dataset.ipynb
    │   ├── test.csv                # obtained by data/prepare_dataset.ipynb
    │   └── subtasks                
    │       ├── age                 # Source: NOTEEVENTS.csv
    │       │   └── ...             # obtained by data/prepare_subtasks.ipynb
    │       ├── sex                 # Source: NOTEEVENTS.csv
    │       │   └── ...             # obtained by data/prepare_subtasks.ipynb
    │       └── diseases            # Source: NOTEEVENTS.csv, D_ICD_DIAGNOSES.csv
    │           └── ...             # obtained by data/prepare_subtasks.ipynb
    └── vocab               
       ├── special_tokens_map.json # obtained by data/prepare_dataset.ipynb
       ├── tokenizer_config.json   # obtained by data/prepare_dataset.ipynb
       └── vocab.txt               # obtained by data/prepare_dataset.ipynb
```
We also provide insturctions to re-produce our data preparation process in [data/README.md](https://github.com/AI-in-Hospitals/Patient-Instructions/tree/master/data).

## Pretreatments
Run the following codes to prepare some necessary files:
```
# Generate the adjacent matrix of all unique digonosis, medication, and procedure codes
# This is essential if we want to use knowledge graph to assist PI-Writer
python pretreatments/prepare_codes_adjacent_matrix.py

# Get top-300 most similar admission records from the training set for each query hospital admission
# This is essential if we want to retrieve historical PIs to assist PI-Writer
python pretreatments/prepare_relevant_info.py

# Use bert-base-uncased to extract sentence-level embeddings of PIs
# We apply max pooling on the word embs of the last layer
python pretreatments/extract_instruction_embs.py 
```


## Training
Here are some key argument to run `train.py`:
- `gpus`: specify the number of gpus;
- `batch_size`: specify the number of samples in a batch;
- `accumulate_grad_batches`: use it if you don't have much gpu memory;
- `arch`: specify the architecture, can be either `small` (hidden size = 256) or `base` (hidden size = 512). See [configs/archs](https://github.com/AI-in-Hospitals/Patient-Instructions/tree/master/config/archs);
- `setup`: specify which setup to use. See options in [config/setups.yaml](https://github.com/AI-in-Hospitals/Patient-Instructions/tree/master/config/setups.yaml), where we provide setups for model variants such as Transformer-based `transformer` and `transformer_Full` and LSTM-based `lstm` and `lstm_Full`.

Here are some examples:
```
python train.py --gpus 8 --batch_size 8 --arch base --setup transformer
python train.py --gpus 8 --batch_size 8 --arch base --setup transformer_Full
python train.py --gpus 8 --batch_size 4 --accumulate_grad_batches 2 --arch base --setup transformer_Full

python train.py --gpus 8 --batch_size 8 --arch small --setup lstm
python train.py --gpus 8 --batch_size 8 --arch small --setup lstm_Full
```

## Evaluation
1. The simplest command below can show you results of automatic metrics (`Bleu`, `METEOR`, and `ROUGE`), which will be written to `./csv_results/overall.csv`.
```
python translate.py $path_to_model
```

2. You can save the generated patient instructions by running:
```
# The ouput file will be saved to `./inference_results/preds_and_scores.json` in this case
python translate.py $path_to_model --save_json --save_base_path ./inference_results --save_folder "" --json_file_name preds_and_scores.json 
```

3. You can evaluate the model on subtasks (see [data/README.md](https://github.com/AI-in-Hospitals/Patient-Instructions/tree/master/data) for details) by passing the augment `--subtask_type`:
```
python translate.py $path_to_model --subtask_type age
python translate.py $path_to_model --subtask_type sex
python translate.py $path_to_model --subtask_type disease
```


## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email {yangbang@pku.edu.cn, fenglinliu98@pku.edu.cn}. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please consider citing our papers if our code or datasets are useful to your work, thanks sincerely!

```bibtex
@inproceedings{liu2022retrieve,
   title={Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions},
   author={Liu, Fenglin and Yang, Bang and You, Chenyu and Wu, Xian and Ge, Shen and Liu, Zhangdaihong and Sun, Xu and Yang, Yang and Clifton, David A.},
   booktitle={Advances in Neural Information Processing Systems},
   year={2022}
}
```

## Acknowledgements
We borrow some codes from [Shivanandroy/simpleT5](https://github.com/Shivanandroy/simpleT5).
