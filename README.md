# [AAAI2024]BOK-VQA
BOK-VQA : Bilingual Outside Knowledge-based Visual Question Answering via Graph Representation Pretraining

Paper Link : https://arxiv.org/abs/2401.06443

## BOK-VQA Dataset
BOK-VQA dataset comprising 17,836 samples and 282,533 knowledge triples. Each sample contained of an image, question, answer, and $k$ external knowledge IDs that are necessary to solve a question.

we assembled 282,533 triple knowledge entries comprising 1,579 objects and 42 relations from English ConceptNet and DBpedia. The selection criteria for the objects and relations were principally based on the 500 objects and 10 relations used in the FVQA dataset. In addition, considering the usage frequency, we incorporated 1,079 objects derived from ImageNet and supplemented 32 additional relations.

- Dataset Sample
<p align="center">
<img src="https://github.com/mjkmain/BOK-VQA/assets/72269271/9bad52d7-9ca6-4d77-87df-4482cb7267e3" alt="BOK-VQA data sample" width="500"/>
</p>


## GEL-VQA Model architecture
(GEL-VQA : Graph-Embeded Learning-based Visual Question Answering.)
In the context of VQA that uses external knowledge, it is unrealistic to assume that one possesses external knowledge pertaining to the given images and questions. Consequently, we proposed the GEL-VQA model that employs a multitask learning approach to perform triple prediction and uses the predicted triples as 
external knowledge.

<p align="center">
<img src="https://github.com/mjkmain/BOK-VQA/assets/72269271/fcc7f28a-c022-40f5-96d2-43657a09a021" alt="GEL-VQA" width="600"/>
</p>



## Experiment results

<p align="center">
<img src="https://github.com/mjkmain/BOK-VQA/assets/72269271/d8184ebc-6dd6-4bad-96ea-efc7a6232928" alt="results" width="700"/>
</p>

# Training & Test Code

## 1. Environmental settings
### 1.1 Clone this repository 

```bash
git clone https://github.com/mjkmain/BOK-VQA.git
cd BOK-VQA
```

### 1.2 Install packages
```bash
python3 -m venv [env_name]
source [env_name]/bin/activate
```

```bash
pip install -e .
```

### 1.3 Download Images

> You can find our dataset at [AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71357)

After the download is complete, place the image directory inside the data directory.

Your directory structure will then like following:
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
Also, you can find the preprocessed CSV data in the `data` directory.

- **all_triple.csv** : The entire knowledge base consisting of 282,533 triples.
- **BOKVQA_data_en.csv**: English BOKVQA data for training.
- **BOKVQA_data_test_en.csv**: English BOKVQA data for testing.
- **BOKVQA_data_ko.csv**: Korean BOKVQA data for training.
- **BOKVQA_data_test_ko.csv**: Korean BOKVQA data for testing.

## 2. Train KGE 
> At `KGE-train` directory,

First, you need to train the KGE before training the VQA model.

```bash
python kge_convkb_train.py 
```

When the end of the training, you'll find the saved files in the `kge_save` directory.

**You need to change the `KGE_DIR` and `DATA_DIR` path in the `util_functions.py` and `IMAGE_DIR` path in the `vqa_datasets.py`**

## 3. Train the VQA model
> At `train` directory,

* To train the GEL-VQA model, use the following command:
```bash
python train_GEL-VQA.py --lang ['ko', 'en', 'bi'] 
```

### arguments
- `--lang`: Selects the language for training.
  - `ko`: Korean
  - `en`: English
  - `bi`: Bilingual (both of English and Korean)

Make sure to replace [ko, en, bi] with your choice of language. For example, if you wish to train on English data, your command would be: 
```bash
python train_GEL-VQA_Ideal.py --lang en 
```

After training, you can find the saved VQA model file in the saved_model directory.

## 4. Test
> At `test` directory,

* To test the GEL-VQA model, use the following command:
```bash
python test_GEL-VQA.py --file_name [FILENAME] --lang ['ko', 'en', 'bi']
```

The `file_name` is organized as follows:

    [model_name]_[lang]_[accuracy].pt

### Contact
mjkmain@seoultech.ac.kr
sswoo@seoultech.ac.kr
