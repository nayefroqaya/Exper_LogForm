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

# ============================================================
# Cross-dataset supervised LogFormer: BGL -> HDFS
# ============================================================

# 1. Preprocess BGL and HDFS independently
# No change in this python file for normal only
python preprocess/preprocess_normalandanomaly_dataset_selector_from_config_no_overlap_last.py \
  --config preprocess/config_cross_dataset_last.yml


# 2. Pre-train LogFormer on SOURCE BGL only
# No change in this python file for normal only
python Fraktion_normaandanomal_train_transformer_pkl_ready_last.py \
  --config preprocess/config_cross_dataset_last.yml


# 3. Tune the pretrained BGL model on TARGET HDFS
#    Uses the configured fraction of HDFS train:
#    normal + anomaly
# there is chnage in pythoin here 
python Frackition_normalandanomal_tune_transformer_pkl_ready_last.py \
  --config preprocess/config_cross_dataset_last.yml


