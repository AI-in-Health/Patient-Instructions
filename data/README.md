# Data Preparation

Code for our paper published in NeurIPS 2022 [[arXiv]](https://arxiv.org/abs/2210.12777)]:
> **Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions**
> 
> Fenglin Liu, Bang Yang, Chenyu You, Xian Wu, Shen Ge, Zhangdaihong Liu, Xu Sun*, Yang Yang*, and David A. Clifton.

## Updates
`[2022.10.25]` We release our data at [Google Drive](https://drive.google.com/file/d/1z1SvPDZ_yixuWuzQr9aK7bNPJUq2tEhY/view?usp=sharing). The data looks like
```
data
├── diagnose-procedure-medication
│   ├── admDxMap_mimic3.pk      # Source: D_ICD_DIAGNOSES.csv
│   ├── admMedMap_mimic3.pk     # Source: prescriptions.csv
│   ├── admPrMap_mimic3.pk      # Source: D_ICD_PROCEDURES.csv
│   └── readme.txt
├── splits                      # Source: NOTEEVENTS.csv
│   ├── train.csv               # obtained by prepare_dataset.ipynb
│   ├── val.csv                 # obtained by prepare_dataset.ipynb
│   ├── test.csv                # obtained by prepare_dataset.ipynb
│   └── subtasks                
│       ├── age                 # Source: NOTEEVENTS.csv
│       │   └── ...             # obtained by prepare_subtasks.ipynb
│       ├── sex                 # Source: NOTEEVENTS.csv
│       │   └── ...             # obtained by prepare_subtasks.ipynb
│       └── diseases            # Source: NOTEEVENTS.csv, D_ICD_DIAGNOSES.csv
│           └── ...             # obtained by prepare_subtasks.ipynb
└── vocab                       
    ├── special_tokens_map.json # obtained by prepare_dataset.ipynb
    ├── tokenizer_config.json   # obtained by prepare_dataset.ipynb
    └── vocab.txt               # obtained by prepare_dataset.ipynb
```

## Directly Use Our Released Data for Patient Instruction Generation
1. Clone the repo: `git clone https://github.com/AI-in-Hospitals/Patient-Instructions.git`
2. For clarity, we define `DATA_ROOT` by running `export DATA_ROOT=$(pwd)/Patient-Instructions/data`
3. Download our released dataset (`data.zip`, 132MB), unzip it and move evering in the data folder to `DATA_ROOT`:
```
unzip data.zip
mv data/* $DATA_ROOT
```


## Reproduce Our Data Preparation Process: Preliminaries
1. Visit the official website of [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), follow guidelines and download the raw dataset (a `.zip` file, ~6.6G)
2. Unzip the downloaded archive file, you should see the following structure:
```
.
└── mimic-iii-clinical-database-1.4
    ├── ADMISSIONS.csv.gz
    ├── CALLOUT.csv.gz
    ├── CAREGIVERS.csv.gz
    ├── CHARTEVENTS.csv.gz
    ├── CPTEVENTS.csv.gz
    ├── DATETIMEEVENTS.csv.gz
    ├── DIAGNOSES_ICD.csv.gz
    ├── DRGCODES.csv.gz
    ├── D_CPT.csv.gz
    ├── D_ICD_DIAGNOSES.csv.gz
    ├── D_ICD_PROCEDURES.csv.gz
    ├── D_ITEMS.csv.gz
    ├── D_LABITEMS.csv.gz
    ├── ICUSTAYS.csv.gz
    ├── INPUTEVENTS_CV.csv.gz
    ├── INPUTEVENTS_MV.csv.gz
    ├── LABEVENTS.csv.gz
    ├── LICENSE.txt
    ├── MICROBIOLOGYEVENTS.csv.gz
    ├── NOTEEVENTS.csv.gz
    ├── OUTPUTEVENTS.csv.gz
    ├── PATIENTS.csv.gz
    ├── PRESCRIPTIONS.csv.gz
    ├── PROCEDUREEVENTS_MV.csv.gz
    ├── PROCEDURES_ICD.csv.gz
    ├── README.md
    ├── SERVICES.csv.gz
    ├── SHA256SUMS.txt
    ├── TRANSFERS.csv.gz
    ├── checksum_md5_unzipped.txt
    └── checksum_md5_zipped.txt
```
3. Run the code below to obtain `NOTEEVENTS.csv` and `D_ICD_DIAGNOSES.csv` and move them to `DATA_ROOT`
```
gzip -d ./mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz
mv NOTEEVENTS.csv $DATA_ROOT

gzip -d ./mimic-iii-clinical-database-1.4/D_ICD_DIAGNOSES.csv.gz
mv D_ICD_DIAGNOSES.csv $DATA_ROOT
```


## Prepare Dataset

Run `prepare_dataset.ipynb`, where we provide step-by-step instructions. After that, you should see the following structure:
```
$DATA_ROOT
├── subjects_with_PI
│   └── ...
├── patient_instructions
│   └── ...
├── health_records
│   └── ...
├── processed_patient_instructions
│   └── ...
├── processed_health_records
│   └── ...
├── info
│   └── ...
├── splits
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── vocab
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

## Prepare Subtasks
Besides evaluating on the full test set, we also divide the test set into different groups based on `age`, `sex`, and `diseases`. Run `prepare_subtasks.ipynb`, and you will see:
```
$DATA_ROOT
├── splits
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
|   └── subtasks                
│       ├── age                 
│       │   └── test
│       │       ├── 0_55.txt    # age between [0, 55)
│       │       ├── 55_70.txt   # age between [55, 70)
│       │       └── 70_200.txt  # age between [70, 200)
│       ├── sex                 
│       │   └── test
│       │       ├── f.txt       # female
│       │       └── m.txt       # male
│       └── diseases            
│           └── test            # higher the rank, more frequent the disease
│               ├── D_250.txt   # rank 5: Diabetes mellitus   
│               ├── D_272.txt   # rank 2: Hyperlipidemia          
│               ├── D_276.txt   # rank 6: Acidosis          
│               ├── D_285.txt   # rank 7: Anemia          
│               ├── D_401.txt   # rank 1: Hypertension (most frequent)        
│               ├── D_414.txt   # rank 4: Coronary atherosclerosis of native coronary artery          
│               ├── D_427.txt   # rank 3: Atrial fibrillation          
│               ├── D_428.txt   # rank 8: Congestive heart failure          
│               ├── D_518.txt   # rank 9: Acute respiratory failure          
│               └── D_584.txt   # rank 10: Acute kidney failure          
└── ...
```

## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email yangbang@pku.edu.cn. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please consider citing our papers if our code or datasets are useful to your work, thanks sincerely!

```bibtex
@inproceedings{liu2022retrieve,
   title={Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions},
   author={Liu, Fenglin and Yang, Bang and You, Chenyu and Wu, Xian and Ge, Shen and Liu, Zhangdaihong and Sun, Xu and Yang, Yang and Clifton, David A},
   booktitle={Advances in Neural Information Processing Systems},
   year={2022}
}
```
