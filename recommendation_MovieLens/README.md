# MovieLens Recommendation

## 檔案目錄結構

* training_data/：訓練、測試資料。
  * auc_item/：AUC 測試資料，每一檔案以日期格式 %Y-%m-%d 命名，例如 2019-09-20.json。
  * candidates/：推薦候選清單。
    * movies.csv：候選清單。
    * filter/：使用者評分過的電影，用來過濾候選清單，每一檔案以日期格式 %Y-%m-%d 命名，例如 2019-09-20.json。
  * features/：評分紀錄，每一檔案以日期格式 %Y-%m-%d 命名，例如 2019-09-20.csv。
  * item_info/：Item 資訊。
    * genome_tags.json：電影的 genome tags。
    * genres.json：電影類型。
    * vector
      * item
        * doc2vec：以 doc2vec 模型產生的向量，每一模型目錄以 exp 開頭。
          * exp{no}/：{no} 被置換為實驗編號。
            * item.json：原始向量。
            * item_mean.json：做 mean normalization 的向量。
            * item_min_max.json：做 min-max normalization 的向量。
            * item_standardization.json：做 standardization normalization 的向量。
        * fasttext：以 fasttext 模型產生的向量，每一模型目錄以 exp 開頭。
          * exp{no}/：{no} 被置換為實驗編號。
            * item.json：原始向量。
            * item_mean.json：做 mean normalization 的向量。
            * item_min_max.json：做 min-max normalization 的向量。
            * item_standardization.json：做 standardization normalization 的向量。
        * tfidfWord2vec：以 word2vec 模型搭配 TF-IDF 產生的向量，每一模型目錄以 exp 開頭。
          * exp{no}/：{no} 被置換為實驗編號。
            * item.json：原始向量。
            * item_mean.json：做 mean normalization 的向量。
            * item_min_max.json：做 min-max normalization 的向量。
            * item_standardization.json：做 standardization normalization 的向量。
* config/：設定範例檔。
  * model/：模型設定檔。
    * recommendation/：推薦模型設定檔。
    * vectorizer/：向量模型設定檔。
* models/：模型。
  * content_wmf/：WMF 模型。
  * content_wmf_interaction_difference_ensembler/：WMF with UPETI 模型。
  * din/：DIN 模型。
  * din_interaction_difference_ensembler/：DIN with UPETI 模型。
  * lightfm/：LightFM 模型。
  * lightfm_interaction_difference_ensembler/：LightFM with UPETI 模型。
* vectorizer/：向量模型。
  * item/：Item 模型。
    * models/：模型。
      * doc2vec/：doc2vec 模型。
      * fasttext/：fasttext 模型。
      * tfidfWord2vec/：word2vec 搭配 TF-IDF 模型。
    * training_data/：訓練資料。

---

## Training Data

### auc_item

在日期之前，使用者評分為正向和負向的電影。

JSON 格式

```json
{
  "USER_ID": {
    "neg": [ITEM_ID, ...],
    "pos": [ITEM_ID, ...]
  }
}
```

### candidates

#### movies.csv

CSV 欄位

* movieId：電影 ID。

範例

```csv
movieId
1
```

#### filter

在日期之前，使用者評分過的電影。

JSON 格式

```json
{
  "USER_ID": [ITEM_ID, ...]
}
```

### features

CSV 欄位

* userId：使用者 ID。
* movieId：電影 ID。
* rating：評分。
* timestamp：時間，格式 %Y-%m-%d %H:%M:%S，例如 2019-11-20 00:02:10。

範例

```csv
userId,movieId,rating,timestamp
23259.0,3868.0,4.0,2019-11-20 00:02:10
```

### item_info

#### genome_tags.json

JSON 格式

```json
{
  "ITEM_ID": [
    {
      "tagId": TAG_ID,
      "relevance": RELEVANCE_SCORE
    }
  ]
}
```

#### genres.json

JSON 格式

```json
{
  "ITEM_ID": ["GENRE", ...]
}
```

#### vector - item

JSON 格式

```json
{
  "ITEM_ID": [...]
}
```

---

## 執行

### 環境

* Python 3.6.9
* Python modules：見 requirements.txt

### Item Vector

#### Word Cutting

```
python word_cutter.py --data DATA --output OUTPUT
```

參數

* data：商品資料 CSV 檔。
* output：輸出檔案。

範例

```
python word_cutter.py --data data/ml-25m/movies.csv --output word_cut/item/item
```

#### Vectorizer Training

```
python vectorizer.py --model MODEL --item train --data DATA --output_dir OUTPUT_DIR --config CONFIG
```

參數

* model：模型。
  * doc2vec
  * fasttext
  * tfidfWord2vec
* item：項目，設定 train。
* data：商品 corpus JSON 檔。
* output_dir：輸出目錄。
* config：模型設定檔。

範例

```
python vectorizer.py --model doc2vec --item train --data vectorizer/item/training_data/corpus.json --output_dir vectorizer/item/models/doc2vec/exp1 --config vectorizer/item/models/doc2vec/doc2vec_config_exp1.json
```

#### Item Vector Generation

```
python vectorizer.py --model MODEL --item vectorize --data DATA --output OUTPUT --model_dir MODEL_DIR
```

參數

* model：模型。
  * doc2vec
  * fasttext
  * tfidfWord2vec
* item：項目，設定 vectorize。
* data：商品斷詞 JSON 檔。
* output：輸出檔案。
* model_dir：模型目錄。

範例

```
python vectorizer.py --model doc2vec --item vectorize --data word_cut/item/item.json --output vector/item/doc2vec/exp1/item.json --model_dir vectorizer/item/models/doc2vec/exp1
```

#### Item Vector Normalization

```
python normalizer.py --method METHOD --data DATA --output OUTPUT
```

參數

* method：Normalization。
  * mean：Mean normalization。
  * min_max：Min-max normalization。
  * standardization：Standardization normalization。
* data：商品向量 JSON 檔。
* output：輸出檔案。

範例

```
python normalizer.py --method mean --data vector/item/doc2vec/exp1/item.json --output vector/item/doc2vec/exp1/item_mean.json
```

### Rating on Certain Date

```
python data_process.py --item extract_rating_date --data DATA --output_dir OUTPUT_DIR [--year YEAR]
```

參數

* item：項目，設定 extract_rating_date。
* data：評分紀錄檔。
* output_dir：輸出目錄。
* year：要取出的西元年。若給予，只會取該西元年的評分紀錄。若沒有給，則全部取。

範例

```
python data_process.py --item extract_rating_date --data data/ml-25m/ratings.csv --year 2019 --output_dir training_data/features
```

### Genome Tags Extraction

```
python data_process.py --item extract_genome_tag --data DATA --output OUTPUT [--relevance_lower_bound RELEVANCE_LOWER_BOUND]
```

參數

* item：項目，設定 extract_genome_tag。
* data：genome tags 分數檔。
* output：輸出檔案。
* relevance_lower_bound：相關程度分數下限。若給予，只會取大於等於下限的 tags。若沒有給，則全部取。

範例

```
python data_process.py --item extract_genome_tag --data data/ml-25m/genome-scores.csv --relevance_lower_bound 0.75 --output training_data/item_info/genome_tags.json
```

### Genres Extraction

```
python data_process.py --item extract_genres --data DATA --output OUTPUT
```

參數

* item：項目，設定 extract_genres。
* data：電影資料檔。
* output：輸出檔案。

範例

```
python data_process.py --item extract_genres --data data/ml-25m/movies.csv --output training_data/item_info/genres.json
```

### Seen Movie for Test Date

```
python data_process.py --item extract_seen_movie_for_test_date --data DATA --test_date_data TEST_DATE_DATA --output OUTPUT
```

參數

* item：項目，設定 extract_seen_movie_for_test_date。
* data：評分紀錄檔。
* test_date_data：測試日期的評分紀錄檔。
* output：輸出檔案。

範例

```
python data_process.py --item extract_seen_movie_for_test_date --data data/ml-25m/ratings.csv --test_date_data training_data/features/2019-11-20.csv --output training_data/candidates/filter/2019-11-20.json
```

### AUC Item for Test Date

```
python data_process.py --item extract_auc_item_for_test_date --data DATA --test_date_data TEST_DATE_DATA --output OUTPUT
```

參數

* item：項目，設定 extract_auc_item_for_test_date。
* data：評分紀錄檔。
* test_date_data：測試日期的評分紀錄檔。
* output：輸出檔案。

範例

```
python data_process.py --item extract_auc_item_for_test_date --data data/ml-25m/ratings.csv --test_date_data training_data/features/2019-11-20.csv --output training_data/auc_item/2019-11-20.json
```

### Model Training

```
python train.py --test_date TEST_DATE --config CONFIG --data_dir DATA_DIR --output_dir OUTPUT_DIR [--is_train_for_test]
```

參數

* test_date：測試日期，格式 %Y-%m-%d，例如 2019-09-20。
* config：設定檔。
* data_dir：訓練資料目錄。
* output_dir：輸出目錄。
* is_train_for_test：是否訓練做 test，預設做 validation。

範例

```
python train.py --test_date 2019-11-20 --config models/lightfm/lightfm_config_exp1.json --data_dir training_data --output_dir models/lightfm/exp1/validation/2019-11-20
```

### Evaluation

```
python evaluate.py --item ITEM --test_date TEST_DATE --data_dir DATA_DIR --model_dir MODEL_DIR --output OUTPUT [--top_ns TOP_NS] [--is_pass_data_info]
```

參數

* item：項目。
  * auc
  * recall
* test_date：測試日期，格式 %Y-%m-%d，例如 2019-09-20。
* data_dir：訓練資料目錄。
* model_dir：模型目錄。
* output：輸出檔案。
* top_ns：計算 recall 的哪些名次，多個名次以「,」分隔。用於 item 為 recall。
* is_pass_data_info：是否要傳資料資訊，用於 DIN with UPETI 模型。

範例

```
python evaluate.py --item recall --test_date 2019-11-20 --data_dir training_data --model_dir models/lightfm/exp1/validation/2019-11-20 --top_ns 10 --output output/lightfm/exp1/validation/2019-11-20/evaluation_recall.json
```

### Result Organization

#### 模型

```
python organize_result.py --data DATA --item model_info --output OUTPUT
```

參數

* data：模型的 model_info.json，多個以「,」分隔。
* item：項目，設定 model_info。
* output：輸出檔案。

範例

```
python organize_result.py --data models/lightfm/exp1/validation/2019-11-14/model_info.json,models/lightfm/exp1/validation/2019-11-15/model_info.json,models/lightfm/exp1/validation/2019-11-16/model_info.json,models/lightfm/exp1/validation/2019-11-17/model_info.json,models/lightfm/exp1/validation/2019-11-18/model_info.json,models/lightfm/exp1/validation/2019-11-19/model_info.json,models/lightfm/exp1/validation/2019-11-20/model_info.json --item model_info --output models/lightfm/exp1/validation/2019-11-20/model_info.json
```

#### 評估結果

```
python organize_result.py --data DATA --item ITEM --output OUTPUT
```

參數

* data：模型的評估結果 JSON 檔，多個以「,」分隔。
* item：項目。
  * evaluation_auc：AUC 的評估結果。
  * evaluation_recall：Recall 的評估結果。
* output：輸出檔案。

範例

```
python organize_result.py --data output/lightfm/exp1/validation/2019-11-14/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-15/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-16/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-17/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-18/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-19/evaluation_recall.json,output/lightfm/exp1/validation/2019-11-20/evaluation_recall.json --item evaluation_recall --output output/lightfm/exp1/validation/evaluation_recall.json
```

---

## 設定檔

目錄：config

### Models

目錄：model

#### Recommendation

目錄：recommendation

##### LightFM

範例：lightfm_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，lightfm_wrapper。
  * class (str)：Python class 名稱，LightFMWrapper。
  * param
    * model：模型參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
    * fit：訓練參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
* data
  * label：Label。
    * rating_lower_bound：評分分數下限。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * item：電影 features。
    * vector：向量。若不使用，則留空。
      * model：模型名稱。
      * exp_no：實驗編號。
      * normalization：Normalization 名稱。
      * scale：向量值 scale。
    * genres：是否使用類型，true 為是，false 為否。
    * genome_tags：genome tags。若不使用，則留空。
      * relevance_lower_bound：相關程度分數下限。

##### LightFM with UPETI

範例：lightfm_interaction_difference_ensembler_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，lightfm_interaction_difference_ensembler_wrapper。
  * class (str)：Python class 名稱，LightFMInteractionDifferenceEnsemblerWrapper。
  * param
    * split_model：每個時間區間的模型。
      * model：模型參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
      * fit：訓練參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
* data
  * label：Label。
    * rating_lower_bound：評分分數下限。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * item：電影 features。
    * vector：向量。若不使用，則留空。
      * model：模型名稱。
      * exp_no：實驗編號。
      * normalization：Normalization 名稱。
      * scale：向量值 scale。
    * genres：是否使用類型，true 為是，false 為否。
    * genome_tags：genome tags。若不使用，則留空。
      * relevance_lower_bound：相關程度分數下限。
  * interaction_difference
    * split：時間區間數量。

##### DIN

範例：din_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，din_wrapper。
  * class (str)：Python class 名稱，DINWrapper。
  * param
    * model：模型參數，[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
    * compile：[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
    * fit：訓練參數，[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
* data
  * duration：時間長度，單位為天。
  * feature
    * user：使用者。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * item：目標電影。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * hist_item：歷史評分的電影。
      * deepctr
        * type：sparse_var_len
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
          * embedding_name 要為 item 的 name。
          * vocabulary_size 設 auto，data loader 會自動算。
    * negative_sample_amount：隨機產生 negative sample 的數量。
    * target：目標電影的 features。
      * feature name 為 item_info_genres
        * feature_name：item_info_genres。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
      * feature name 為 item_info_genome_tags
        * feature_name：item_info_genome_tags。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
    * history：歷史評分的電影。
      * amount：數量。
      * features：Features。
        * feature name 為 item_info_genres，類型
          * feature_name：item_info_genres。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。
        * feature name 為 item_info_genome_tags，genome tag
          * feature_name：item_info_genome_tags。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。

##### DIN with UPETI

範例：din_interaction_difference_ensembler_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，din_interaction_difference_ensembler_wrapper。
  * class (str)：Python class 名稱，DINInteractionDifferenceEnsemblerWrapper。
  * param
    * split_model：每個時間區間的模型。
      * model：模型參數，[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
      * compile：[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
      * fit：訓練參數，[參見](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.din.html)。
* data
  * duration：時間長度，單位為天。
  * feature
    * user：使用者。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * item：目標電影。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * hist_item：歷史評分的電影。
      * deepctr
        * type：sparse_var_len
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
          * embedding_name 要為 item 的 name。
          * vocabulary_size 設 auto，data loader 會自動算。
    * negative_sample_amount：隨機產生 negative sample 的數量。
    * target：目標電影的 features。
      * feature name 為 item_info_genres
        * feature_name：item_info_genres。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
      * feature name 為 item_info_genome_tags
        * feature_name：item_info_genome_tags。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
    * history：歷史評分的電影。
      * amount：數量。
      * features：Features。
        * feature name 為 item_info_genres，類型
          * feature_name：item_info_genres。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。
        * feature name 為 item_info_genome_tags，genome tag
          * feature_name：item_info_genome_tags。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。
    * interaction_difference
      * split：時間區間數量。

##### content WMF

範例：content_wmf_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，content_wmf_wrapper。
  * class (str)：Python class 名稱，ContentWMFWrapper。
  * param
    * model：模型參數。
      * num_factors：維度。
      * lambda_V_reg：V reg。
      * lambda_U_reg：U reg。
      * lambda_W_reg：W reg。
      * init_std：init std。
      * beta：beta。
      * num_iters：Iteration 數。
      * batch_size：Batch size。
      * random_state：Random state。
      * dtype：資料型態。
      * n_jobs：Jobs 數。
* data
  * label：Label。
    * rating_lower_bound：評分分數下限。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * item：電影 features。
    * vector：向量。若不使用，則留空。
      * model：模型名稱。
      * exp_no：實驗編號。
      * normalization：Normalization 名稱。
      * scale：向量值 scale。
    * genres：是否使用類型，true 為是，false 為否。
    * genome_tags：genome tags。若不使用，則留空。
      * relevance_lower_bound：相關程度分數下限。
  * feature
    * log：調整偏好度。
      * alpha：Alpha。
      * epsilon：Epsilon。

##### content WMF with UPETI

範例：content_wmf_interaction_difference_ensembler_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，content_wmf_interaction_difference_ensembler_wrapper。
  * class (str)：Python class 名稱，ContentWMFInteractionDifferenceEnsemblerWrapper。
  * param
    * split_model：每個時間區間的模型。
      * model：模型參數。
        * num_factors：維度。
        * lambda_V_reg：V reg。
        * lambda_U_reg：U reg。
        * lambda_W_reg：W reg。
        * init_std：init std。
        * beta：beta。
        * num_iters：Iteration 數。
        * batch_size：Batch size。
        * random_state：Random state。
        * dtype：資料型態。
        * n_jobs：Jobs 數。
* data
  * label：Label。
    * rating_lower_bound：評分分數下限。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * item：電影 features。
    * vector：向量。若不使用，則留空。
      * model：模型名稱。
      * exp_no：實驗編號。
      * normalization：Normalization 名稱。
      * scale：向量值 scale。
    * genres：是否使用類型，true 為是，false 為否。
    * genome_tags：genome tags。若不使用，則留空。
      * relevance_lower_bound：相關程度分數下限。
  * feature
    * log：調整偏好度。
      * alpha：Alpha。
      * epsilon：Epsilon。
  * interaction_difference
    * split：時間區間數量。

#### Vectorizer

目錄：vectorizer

##### doc2vec

範例：doc2vec_config_exp0.json

參數：[參見]（https://radimrehurek.com/gensim/models/doc2vec.html）

##### fasttext

範例：fasttext_config_exp0.json

參數：[參見]（https://radimrehurek.com/gensim/models/fasttext.html）

##### word2vec with TF-IDF

範例：tfidfWord2vec_config_exp0.json

參數：[參見]（https://radimrehurek.com/gensim/models/word2vec.html）
