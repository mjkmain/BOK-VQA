# [AAAI2024] BOK-VQA
BOK-VQA : Bilingual Outside Knowledge-based Visual Question Answering via Graph Representation Pretraining

# BOK-VQA

## Overview

You can find the preprocessed CSV data in the `data` directory.

- **all_triple.csv** : The entire knowledge base consisting of 282,533 triples.
- **BOKVQA_data_en.csv**: English BOKVQA data for training.
- **BOKVQA_data_test_en.csv**: English BOKVQA data for testing.

- **BOKVQA_data_ko.csv**: Korean BOKVQA data for training.
- **BOKVQA_data_test_ko.csv**: Korean BOKVQA data for testing.


## 1. Download the image file.

You can download the image files via [G-drive](https://drive.google.com/file/d/1SpOntv2ZIwyNW-JghUc7myJkC9PLs4_H/view?usp=drive_link) or [AI-hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71357)

After the download is complete, place the image directory inside the data directory.

Your directory structure will then look like this:
```
┗━━━ data
      ┣━━━ image
      ┃     ┗━━━ 121100220220707140119.jpg
      ┃     ┗━━━ 121100220220707140304.jpg
      ┃     ┗━━━ 121100520220830104341.jpg
      ┃     ┗━━━ ...
      ┗━━━ all_triple.csv
      ┗━━━ BOKVQA_data_en.csv
      ┗━━━ BOKVQA_data_ko.csv
      ┗━━━ BOKVQA_data_test_en.csv
      ┗━━━ BOKVQA_data_test_ko.csv
```

## 2. Installation all requirements.

```bash
pip install -r requirements.txt
```

## 3. Train KGE 

First, we need to train the KGE before training the VQA model.

```bash
python kge_convkb_train.py 
```

When the end of the training, you'll find the saved files in the `kge_save` directory.

## 4. Train the VQA model

* To train the BASELINE model, use the following command:

```bash
python train_baseline.py --lang ['ko', 'en', 'bi'] --fold [1-5]
```

* To train the GEL-VQA-Ideal model, use the following command:
```bash
python train_GEL-VQA_Ideal.py --lang ['ko', 'en', 'bi'] -- fold [1-5]
```

* To train the GEL-VQA model, use the following command:
```bash
python train_GEL-VQA.py --lang ['ko', 'en', 'bi'] -- fold [1-5]
```

* To train the GEL-VQA-TF model, use the following command:
```bash
python train_GEL-VQA-TF.py --lang ['ko', 'en', 'bi'] -- fold [1-5]
```

* To train the GEL-VQA-TF-ATTN model, use the following command:
```bash
python train_GEL-VQA-TF-ATTN.py --lang ['ko', 'en', 'bi'] -- fold [1-5]
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

## 5. Test

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
