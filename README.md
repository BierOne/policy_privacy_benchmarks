# policy_privacy_benchmarks
This is a pytorch implementation for state-of-the-art results on policy privacy datasets (OPP-115).
The older implementation can be found at this [link](https://github.com/SmartDataAnalytics/Polisis_Benchmark)
. Thanks for her generously sharing the code.

By the way, I just implemented a RNN model for the prediction so far. The update would be coming soon.

## Results

Scores per label with 0.5 threshold (LSTM model)

|Label                     |                      Num      |      F1      |       Precision   |   Recall|
|:----                     |:------------------------------:| :-----------: |:------------------:|:--------:| 
First Party Collection/Use       |               513      |      0.77     |     0.75      |     0.79
Third Party Sharing/Collection   |               400      |      0.77     |      0.78    |       0.77
User Access, Edit and Deletion   |               78        |     0.58     |      0.64     |      0.54
Data Retention                   |               56       |      0.3      |      0.52     |      0.21
Data Security                    |               103      |      0.65     |      0.69    |       0.61
International and Specific Audiences   |         114      |      0.82     |      0.82     |      0.82
Do Not Track                         |           12        |     0.5      |      1.0      |      0.33
Policy Change                     |              62       |      0.72     |      0.73     |      0.71
User Choice/Control                |             213      |      0.63      |     0.59      |     0.66
Introductory/Generic               |             258       |     0.61     |      0.65      |     0.58
Practice not covered               |             220      |      0.42     |      0.46     |     0.39
Privacy contact information        |             101      |      0.67      |     0.7       |     0.64
Micro_average                      |             -        |      0.68     |      0.69      |     0.66
Macro_average                    |               -         |     0.62     |      0.7       |     0.59


## Prerequisites
- python3.7 is suggested, though python2 is also compatible
- pytorch (v1.4 or higher version)
- tqdm, json, numpy

## Preprocessing
1. Preprocess raw data. (refer to this [link](https://github.com/SmartDataAnalytics/Polisis_Benchmark/tree/master/raw_data) )
    ```
    python tools/preprocess_raw_data.py
    ```
2. Create dictionary and embeddings for all tokens in datasets.
    ```
    python create_dictionary.py
    ```
3. Split datasets into train (75 policies) and test (40 policies).
    ```
    python tools/split_datasets.py
    ```

## Training

```
python main.py --gpu 0
```
