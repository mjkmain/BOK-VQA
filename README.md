# [AAAI2024]BOK-VQA
BOK-VQA : Bilingual Outside Knowledge-based Visual Question Answering via Graph Representation Pretraining

## BOK-VQA Dataset
BOK-VQA dataset comprising 17,836 samples and 282,533 knowledge triples. Each sample contained of an image, question, answer, and $k$ external knowledge IDs that are necessary to solve a question.

we assembled 282,533 triple knowledge entries comprising 1,579 objects and 42 relations from English ConceptNet and DBpedia. The selection criteria for the objects and relations were principally based on the 500 objects and 10 relations used in the FVQA dataset. In addition, considering the usage frequency, we incorporated 1,079 objects derived from ImageNet and supplemented 32 additional relations.

## GEL-VQA Model architecture
(GEL-VQA : Graph-Embeded Learning-based Visual Question Answering.)
In the context of VQA that uses external knowledge, it is unrealistic to assume that one possesses external knowledge pertaining to the given images and questions. Consequently, we proposed the GEL-VQA model that employs a multitask learning approach to perform triple prediction and uses the predicted triples as 
external knowledge.
![GEL-VQA](https://github.com/mjkmain/BOK-VQA/assets/72269271/fcc7f28a-c022-40f5-96d2-43657a09a021)

## Experiment results
![image](https://github.com/mjkmain/BOK-VQA/assets/72269271/d8184ebc-6dd6-4bad-96ea-efc7a6232928)


# Training & Test Code



## Overview

> You can find our dataset at https://huggingface.co/datasets/mjkmain/bok-vqa-dataset. 



## 1. Installation all requirements.


```bash
pip install -r requirements.txt
```

`git clone` and add python path via

`export PYTHONPATH="$(pwd):$PYTHONPATH"` in the BOK-VQA directory.

## 2. Train KGE 

First, we need to train the KGE before training the VQA model.

```bash
python kge_convkb_train.py 
```

When the end of the training, you'll find the saved files in the `kge_save` directory.

**You need to change the `kge_dir` path in the `util_functions.py' **

## 3. Train the VQA model

* To train the BASELINE model, use the following command:

```bash
python train_baseline.py --lang ['ko', 'en', 'bi'] 
```

* To train the GEL-VQA-Ideal model, use the following command:
```bash
python train_GEL-VQA_Ideal.py --lang ['ko', 'en', 'bi'] 
```

* To train the GEL-VQA model, use the following command:
```bash
python train_GEL-VQA.py --lang ['ko', 'en', 'bi'] 
```

* To train the GEL-VQA-TF model, use the following command:
```bash
python train_GEL-VQA-TF.py --lang ['ko', 'en', 'bi'] 
```

* To train the GEL-VQA-TF-ATTN model, use the following command:
```bash
python train_GEL-VQA-TF-ATTN.py --lang ['ko', 'en', 'bi'] 
```

### arguments
- `--lang`: Selects the language for training.
  - `ko`: Korean
  - `en`: English
  - `bi`: Bilingual

Make sure to replace [ko, en, bi] with your choice of language. For example, if you wish to train on English data, your command would be: 
```bash
python train_GEL-VQA_Ideal.py --lang en --fold 1
```

- `--fold`: Determines the validation fold.
  - Value should be an integer between 1 to 5.

For instance, if you wish to train using the third fold, you would use:
```bash
python train_GEL-VQA_Ideal.py --lang ko --fold 3
```

The default value of `fold` is 1

After training, you can find the saved VQA model file in the saved_model directory.

## 4. Test

* To test the BASELINE model, use the following command:
```bash
python test_baseline.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

* To test the GEL-VQA-Ideal model, use the following command:
```bash
python test_GEL-VQA-Ideal.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

* To test the GEL-VQA model, use the following command:
```bash
python test_GEL-VQA.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

* To test the GEL-VQA-TF model, use the following command:
```bash
python test_GEL-VQA.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

* To test the GEL-VQA-TF-ATTN model, use the following command:
```bash
python test_GEL-VQA-TF-ATTN.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

**NOTE** : The GEL-VQA model and the GEL-VQA-TF model use the same test file.

The `file_name` is organized as follows:

    [model_name]_[lang]_[fold]_[accuracy].pt

The `model_name` corresponds to:

* BASE : BASELINE model
* GEL-Ideal : GEL-VQA-Ideal model
* GEL : GEL-VQA model
* GEL-TF : GEL-VQA-TF model
* GEL-TF-ATTN : GEL-VQA-TF-ATTN model

### Contact
mjkmain@seoultech.ac.kr
