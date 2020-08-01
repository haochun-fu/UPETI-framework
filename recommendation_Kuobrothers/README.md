# Kuobrothers Recommendation

## 檔案目錄結構

* training_data/：訓練、測試資料。
  * food123/：好吃市集。
    * available_product/：架上商品清單，每一檔案以日期格式 %Y-%m-%d 命名，例如 2019-09-19.csv。
    * features/：使用者與商品的互動紀錄，每一檔案以日期格式 %Y-%m-%d 命名，例如 2019-09-19.csv。
    * product_info/：商品資訊。
      * product_info.json
    * product_vector/：商品向量。
      * doc2vec/：以 doc2vec 模型產生的向量。
        * size_{dim}/：dim 維度的向量。{dim} 被置換為維度數值。
          * product.json：原始向量。
          * product_mean.json：做 mean normalization 的向量。
          * product_min_max.json：做 min-max normalization 的向量。
          * product_standardization.json：做 standardization normalization 的向量。
      * fasttext/：以 fasttext 模型產生的向量。
        * size_{dim}/：dim 維度的向量。{dim} 被置換為維度數值。
          * product.json：原始向量。
          * product_mean.json：做 mean normalization 的向量。
          * product_min_max.json：做 min-max normalization 的向量。
          * product_standardization.json：做 standardization normalization 的向量。
      * tfidfWord2vec/：以 word2vec 模型搭配 TF-IDF 產生的向量。
        * size_{dim}/：dim 維度的向量。{dim} 被置換為維度數值。
          * product.json：原始向量。
          * product_mean.json：做 mean normalization 的向量。
          * product_min_max.json：做 min-max normalization 的向量。
          * product_standardization.json：做 standardization normalization 的向量。
* config/：設定範例檔。
  * model/：模型設定檔。
    * recommendation/：推薦模型設定檔。
    * vectorizer/：向量模型設定檔。
* models/：模型。
  * content_wmf/：WMF 模型。
    * food123/：好吃市集。
  * content_wmf_interaction_difference_ensembler/：WMF with UPETI 模型。
    * food123/：好吃市集。
  * content_wmf_interaction_difference_merger/：WMF with method 2 of UPETI extenion 模型。
    * food123/：好吃市集。
  * content_wmf_interaction_difference_merger_v2/：WMF with method 1 of UPETI extension 模型。
    * food123/：好吃市集。
  * din/：DIN 模型。
    * food123/：好吃市集。
  * din_interaction_difference_ensembler/：DIN with UPETI 模型。
    * food123/：好吃市集。
  * lightfm/：LightFM 模型。
    * food123/：好吃市集。
  * lightfm_interaction_difference_ensembler/：LightFM with UPETI 模型。
    * food123/：好吃市集。
  * lightfm_interaction_difference_merger/：LightFM with method 2 of UPETI extenion 模型。
    * food123/：好吃市集。
  * lightfm_interaction_difference_merger_v2/：LightFM with method 1 of UPETI extension 模型。
    * food123/：好吃市集。
* vectorizer/：向量模型。
  * doc2vec/：doc2vec 模型。
  * fasttext/：fasttext 模型。
  * tfidfWord2vec/：word2vec 搭配 TF-IDF 模型。

---

## Training Data

### available_product

CSV 欄位

* contractid：商品 ID。

### features

CSV 欄位

* user_id：使用者 ID。
* contract_id：商品 ID。
* convert：是否購買，1：是，0：不是。
* favorite：是否收藏，1：是，0：不是。
* pageview：是否點擊觀看，1：是，0：不是。
* ref_search：是否來自搜尋，1：是，0：不是。
* time：時間，格式「%Y-%m-%d %H:%M:%S.%f」，例如 2019-09-19 00:00:02.132。

範例

```csv
user_id,contract_id,convert,favorite,pageview,ref_search,time
316890,190608,0,0,1,0,2019-09-19 00:00:02.132
```

### product_info

JSON 格式

```json
{
  "CONTRACT_ID": {
    "category": [...],
    "tag": [...],
    "keyword": [...]
  }
}
```

每一項目的值依值於該項目的出現次數由多排到少。

範例
```json
{
  "138161": {
    "category": ["居家生活雜貨", "環保袋/飲料提袋"],
    "tag": ["飲料袋"],
    "keyword": ["環保飲料提袋", "環保杯套", "飲料提袋", "杯套", "飲料袋"]
  }
}
```

### product_vector

JSON 格式

```json
{
  "CONTRACT_ID": {
    "name": [...],
    "desc_short": [...]
  }
}
```

---

## 執行

### 環境

* Python 3.6.9
* Python modules：見 requirements.txt

### Word Cutting

```
python word_cutter.py --data DATA --output OUTPUT
```

參數

* data：商品資料 CSV 檔。
* output：輸出檔案。

範例

```
python word_cutter.py --data data/product_info/product_info_k_products.csv --output product_word_cut/product
```

### Vectorizer Training

```
python vectorizer.py --model MODEL --item train --data DATA --output OUTPUT --config CONFIG
```

參數

* model：模型。
  * doc2vec
  * fasttext
  * tfidfWord2vec
* item：項目，設定 train。
* data：商品 corpus JSON 檔。
* output：輸出檔案。
* config：模型設定檔。

範例

```
python vectorizer.py --model doc2vec --item train --data product_word_cut/product_corpus.json --output vectorizer/doc2vec/exp1/model --config vectorizer/doc2vec/doc2vec_config_exp1.json
```

### Product Vector Generation

```
python vectorizer.py --model MODEL --item vectorize --data DATA --output OUTPUT --model_file MODEL_FILE
```

參數

* model：模型。
  * doc2vec
  * fasttext
  * tfidfWord2vec
* item：項目，設定 vectorize。
* data：商品斷詞 JSON 檔。
* output：輸出檔案。
* model_file：模型檔。

範例

```
python vectorizer.py --model doc2vec --item vectorize --data product_word_cut/product.json --output training_data/food123/product_vector/doc2vec/size_64/product.json --model_file vectorizer/doc2vec/exp1/model
```

### Product Vector Normalization

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
python normalizer.py --method mean --data training_data/food123/product_vector/doc2vec/size_64/product.json --output training_data/food123/product_vector/doc2vec/size_64/product_mean.json
```

### Model Training

```
python train.py --test_date TEST_DATE --config CONFIG --data_dir DATA_DIR --output_dir OUTPUT_DIR [--is_train_for_test] [--ensembler_model_dir]
```

參數

* test_date：測試日期，格式 %Y-%m-%d，例如 2019-09-19。
* config：設定檔。
* data_dir：訓練資料目錄。
* output_dir：輸出目錄。
* is_train_for_test：是否訓練做 test，預設做 validation。
* ensembler_model_dir：UPETI 模型目錄，當訓練的模型為 method 2 of UPETI extension 時，可以使用已訓練好的 UPETI 模型。

範例

```
python train.py --test_date 2019-09-19 --config models/lightfm/food123/lightfm_config_exp1.json --data_dir training_data/food123 --output_dir models/lightfm/food123/exp1/validation/2019-09-19
```

### Evaluation

```
python evaluate.py --test_date TEST_DATE --data_dir DATA_DIR --model_dir MODEL_DIR --top_ns TOP_NS --output OUTPUT [--is_pass_data_info]
```

參數

* test_date：測試日期，格式 %Y-%m-%d，例如 2019-09-19。
* data_dir：訓練資料目錄。
* model_dir：模型目錄。
* top_ns：計算 recall 的哪些名次，多個名次以「,」分隔。
* output：輸出檔案。
* is_pass_data_info：是否要傳資料資訊，用於 DIN with UPETI 模型。

範例

```
python evaluate.py --test_date 2019-09-19 --data_dir training_data/food123 --model_dir models/lightfm/food123/exp1/validation/2019-09-19 --top_ns 10 --output output/lightfm/food123/exp1/validation/2019-09-19/evaluation.json
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
python organize_result.py --data models/lightfm/food123/exp1/validation/2019-09-13/model_info.json,models/lightfm/food123/exp1/validation/2019-09-14/model_info.json,models/lightfm/food123/exp1/validation/2019-09-15/model_info.json,models/lightfm/food123/exp1/validation/2019-09-16/model_info.json,models/lightfm/food123/exp1/validation/2019-09-17/model_info.json,models/lightfm/food123/exp1/validation/2019-09-18/model_info.json,models/lightfm/food123/exp1/validation/2019-09-19/model_info.json --item model_info --output models/lightfm/food123/exp1/validation/model_info.json
```

#### 評估結果

```
python organize_result.py --data DATA --item evaluation --output OUTPUT
```

參數

* data：模型的評估結果 JSON 檔，多個以「,」分隔。
* item：項目，設定 evaluation。
* output：輸出檔案。

範例

```
python organize_result.py --data output/lightfm/food123/exp1/validation/2019-09-13/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-14/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-15/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-16/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-17/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-18/evaluation.json,output/lightfm/food123/exp1/validation/2019-09-19/evaluation.json --item evaluation --output output/lightfm/food123/exp1/validation/evaluation.json
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
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * product_info：商品資訊。
    * category：種類使用數量，-1 表示全使用。
    * keyword：關鍵字使用數量，-1 表示全使用。
    * tag：標籤使用數量，-1 表示全使用。

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
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * product_info：商品資訊。
    * category：種類使用數量，-1 表示全使用。
    * keyword：關鍵字使用數量，-1 表示全使用。
    * tag：標籤使用數量，-1 表示全使用。
  * interaction_difference
    * split：時間區間數量。

##### LightFM with method 1 of UPETI extension

範例：lightfm_interaction_difference_merger_v2_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，lightfm_interaction_difference_merger_v2_wrapper。
  * class (str)：Python class 名稱，LightFMInteractionDifferenceMergerV2Wrapper。
  * param
    * model：模型參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
    * fit：訓練參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
* data
  * label：Label。
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * product_info：商品資訊。
    * category：種類使用數量，-1 表示全使用。
    * keyword：關鍵字使用數量，-1 表示全使用。
    * tag：標籤使用數量，-1 表示全使用。
  * interaction_difference
    * split：時間區間數量。
    * time_decay：時間權重。
      * name：名稱，例如 linear 為 linear decay。
      * param：參數，例如 linear 為 delta，表示每天的權重變化量。

##### LightFM with method 2 of UPETI extension

範例：lightfm_interaction_difference_merger_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，lightfm_interaction_difference_merger_wrapper。
  * class (str)：Python class 名稱，LightFMInteractionDifferenceMergerWrapper。
  * param
    * split_model：每個時間區間的模型。
      * model：模型參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
      * fit：訓練參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
    * merger_model：整合模型。
      * model：模型參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
      * fit：訓練參數，[參見](https://making.lyst.com/lightfm/docs/lightfm.html)。
      * data
        * scale：值 scale。
        * filter：過濾。
          * lower_bound：預測值下限。
            * include：大於或等於的值，null 為不使用。
            * exclude：大於的值，null 為不使用。
          * past_user_item_pair：保留 UPETI 模型的訓練資料看過的使用者和商品的關係，soft 為會保留通過其它過濾方法的使用者和商品的關係，hard 為只保留見過的使用者和商品的關係，null 為不使用。。
* data
  * label：Label。
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * product_info：商品資訊。
    * category：種類使用數量，-1 表示全使用。
    * keyword：關鍵字使用數量，-1 表示全使用。
    * tag：標籤使用數量，-1 表示全使用。
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
  * label：Label。
    * action：要使用哪些行為，convert、pageview、favorite。
  * duration：時間長度，單位為天。
  * feature
    * user：使用者。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * item：目標商品。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * hist_item：歷史互動的商品。
      * deepctr
        * type：sparse_var_len
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
          * embedding_name 要為 item 的 name。
          * vocabulary_size 設 auto，data loader 會自動算。
    * negative_sample_amount：隨機產生 negative sample 的數量。
    * target：目標商品的 features。
      * feature name 為 action，行為次數
        * feature_name：action。
        * duration：時間長度，單位為天。
        * action_weight：行為權重。
          * general：所有。
            * convert：購買。
            * favorite：收藏。
            * pageview：點擊觀看。
          * ref_search：來自搜尋。
            * convert：購買。
            * favorite：收藏。
            * pageview：點擊觀看。
        * time_weight：時間權重。
          * name：名稱，例如 decrease_by_day 為 linear decay。
        * deepctr
          * type：dense
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#densefeat）
      * feature name 為 product_info，商品資訊
        * feature_name：product_info。
        * name：資訊名稱，category、tag、keyword。
        * amount：數量設為 1。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
            * vocabulary_size 設 auto+1，data loader 會自動算，+1 留給未知的。
    * history：歷史互動的商品。
      * amount：數量。
      * features：Features。
        * feature name 為 product_info，商品資訊
          * feature_name：product_info。
          * name：資訊名稱，category、tag、keyword。
          * amount：數量設為 1。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。
              * vocabulary_size 設 auto+1，data loader 會自動算，+1 留給未知的。

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
  * label：Label。
    * action：要使用哪些行為，convert、pageview、favorite。
  * duration：時間長度，單位為天。
  * feature
    * user：使用者。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * item：目標商品。
      * deepctr
        * type：sparse
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
          * vocabulary_size 設 auto，data loader 會自動算。
    * hist_item：歷史互動的商品。
      * deepctr
        * type：sparse_var_len
        * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
          * embedding_name 要為 item 的 name。
          * vocabulary_size 設 auto，data loader 會自動算。
    * negative_sample_amount：隨機產生 negative sample 的數量。
    * target：目標商品的 features。
      * feature name 為 action，行為次數
        * feature_name：action。
        * duration：時間長度，單位為天。
        * action_weight：行為權重。
          * general：所有。
            * convert：購買。
            * favorite：收藏。
            * pageview：點擊觀看。
          * ref_search：來自搜尋。
            * convert：購買。
            * favorite：收藏。
            * pageview：點擊觀看。
        * time_weight：時間權重。
          * name：名稱，例如 decrease_by_day 為 linear decay。
        * deepctr
          * type：dense
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#densefeat）
      * feature name 為 product_info，商品資訊
        * feature_name：product_info。
        * name：資訊名稱，category、tag、keyword。
        * amount：數量設為 1。
        * encoding：設定 label 做 label encoding。
        * deepctr
          * type：sparse
          * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#sparsefeat）
            * vocabulary_size 設 auto+1，data loader 會自動算，+1 留給未知的。
    * history：歷史互動的商品。
      * amount：數量。
      * features：Features。
        * feature name 為 product_info，商品資訊
          * feature_name：product_info。
          * name：資訊名稱，category、tag、keyword。
          * amount：數量設為 1。
          * encoding：設定 label，做 label encoding。
          * deepctr
            * type：sparse_var_len
            * param：[參見]（https://deepctr-doc.readthedocs.io/en/latest/Features.html#varlensparsefeat）
              * embedding_name 要為 target 對應 feature 的 name。
              * vocabulary_size 設 auto+1，data loader 會自動算，+1 留給未知的。
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
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
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
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * feature
    * log：調整偏好度。
      * alpha：Alpha。
      * epsilon：Epsilon。
  * interaction_difference
    * split：時間區間數量。

##### content WMF with method 1 of UPETI extension

範例：content_wmf_interaction_difference_merger_v2_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，content_wmf_interaction_difference_merger_v2_wrapper。
  * class (str)：Python class 名稱，ContentWMFInteractionDifferenceMergerV2Wrapper。
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
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
  * feature
    * log：調整偏好度。
      * alpha：Alpha。
      * epsilon：Epsilon。
  * interaction_difference
    * split：時間區間數量。
    * time_decay：時間權重。
      * name：名稱，例如 linear 為 linear decay。
      * param：參數，例如 linear 為 delta，表示每天的權重變化量。

##### content WMF with method 2 of UPETI extension

範例：content_wmf_interaction_difference_merger_config_exp0.json

參數

* model
  * module (str)：Python module 名稱，content_wmf_interaction_difference_merger_wrapper。
  * class (str)：Python class 名稱，ContentWMFInteractionDifferenceMergerWrapper。
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
    * merger_model：整合模型。
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
        * scale：值 scale。
        * filter：過濾。
          * lower_bound：預測值下限。
            * include：大於或等於的值，null 為不使用。
            * exclude：大於的值，null 為不使用。
          * past_user_item_pair：保留 UPETI 模型的訓練資料看過的使用者和商品的關係，soft 為會保留通過其它過濾方法的使用者和商品的關係，hard 為只保留見過的使用者和商品的關係，null 為不使用。
* data
  * label：Label。
    * action_weight：行為權重。
      * general：所有。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
      * ref_search：來自搜尋。
        * convert：購買。
        * favorite：收藏。
        * pageview：點擊觀看。
    * time_weight：時間權重。
      * name：名稱，例如 decrease_by_day 為 linear decay。
      * is_use_first_decay：是否使用者的第一筆紀錄出現後才開始 decay。
  * duration：時間長度，單位為天。
  * product_vector：商品向量。若不使用，留空。
    * model：模型名稱。
    * size：向量維度。
    * normalization：Normalization 名稱。
    * field：欄位。
    * field_operation：欄位整合方式，concatenate 為串接，add 為相加。
    * scale：向量值 scale。
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
