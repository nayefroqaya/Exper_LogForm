# LogFormer
[AAAI 2024] LogFormer: A Pre-train and Tuning Pipeline for Log Anomaly Detection

# Data
Training data can be download from [LogHub](https://github.com/logpai/loghub)


# Updates
01/23. We release the base code version for LogFormer, which is a strong baseline for log anomaly detection.

# Data processing
1. Downloading data into log_data/
2. parse_log.py
3. preprocess_xxx.py

# Run
1. First run train_transformer.py
2. Then run tune_transformer.py


# Citation
If you feel helpful, please cite our paper.

```
@inproceedings{guo2024logformer,
  title={Logformer: A pre-train and tuning pipeline for log anomaly detection},
  author={Guo, Hongcheng and Yang, Jian and Liu, Jiaheng and Bai, Jiaqi and Wang, Boyang and Li, Zhoujun and Zheng, Tieqiao and Zhang, Bo and Peng, Junran and Tian, Qi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={135--143},
  year={2024}
}
```


## ----------- Cross dataset running : 
- python preprocess/preprocess_dataset_selector_from_config_no_overlap_last.py --config preprocess/config_cross_dataset_last.yml

- Train cross-dataset model from scratch
  python train_transformer_pkl_ready_last.py \
  --log_name BGL_to_HDFS \
  --window_size 120 \
  --preprocessed_dir preprocess/preprocessed_data

- Tune using pretrained BGL source checkpoint
python tune_transformer_pkl_ready_last.py \
  --log_name BGL_to_HDFS \
  --window_size 120 \
  --preprocessed_dir preprocess/preprocessed_data \
  --pretrained_log_name BGL \
  --load_path checkpoints/train_BGL_classifier_1_64_1e-05-best.pt
- 
 ### --load_path checkpoints/YOUR_SOURCE_MODEL-best.pt
## ----------- In domain running  : 

- python preprocess/preprocess_dataset_selector_from_config_no_overlap_last.py --config preprocess/config_cross_dataset_last.yml

- python train_transformer_pkl_ready_last_paper4.py \
  --log_name HDFS \
  --window_size 120 \
  --preprocessed_dir preprocess/preprocessed_data

- python tune_transformer_pkl_ready_last.py \
  --log_name BGL \
  --window_size 120 \
  --preprocessed_dir preprocess/preprocessed_data \
  --pretrained_log_name random
