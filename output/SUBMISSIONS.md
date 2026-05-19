# ASG3 Submission Tracker

| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |
|------|------|---------------|-------------|-------|----------|-------|
| submission_baseline_rf.csv | May 16 | - | 0.7597 | RF (sklearn) | 42 base | baseline |
| submission_xgb.csv | May 16 | - | 0.7358 | XGBoost | 42 base | 50 trials |
| submission_lgb.csv | May 16 | - | **0.7645** | LightGBM | 42 base | 50 trials |
| submission_ensemble.csv | May 16 | - | 0.7587 | XGB+LGB ensemble | 42 base | soft vote |
| submission_baseline_rf_v2.csv | May 17 | - | ? | RF (sklearn) | 82 (skew,kurt,jerk,SMA,corr) | |
| submission_xgb_v2.csv | May 17 | - | 0.7535 | XGBoost | 82 (skew,kurt,jerk,SMA,corr) | 50 trials, no SMOTE |
| submission_lgb_v2.csv | May 17 | - | 0.7453 | LightGBM | 82 (skew,kurt,jerk,SMA,corr) | 50 trials, no SMOTE |
| submission_ensemble_v2.csv | May 17 | - | ? | XGB+LGB ensemble | 82 (skew,kurt,jerk,SMA,corr) | soft vote |
| submission_xgb_20260518_185105_01.csv | 20260518_185105 | baefa7ef | ? | XGBoost | 42 base | 150 trials, SMOTE inside GroupKFold |
| submission_lgb_20260518_185105_02.csv | 20260518_185105 | df9e5e49 | **0.7653** | LightGBM | 42 base | 150 trials, SMOTE inside GroupKFold |
| submission_ensemble_20260518_185105_03.csv | 20260518_185105 | acbe485b | 0.7575 | XGBoost + LightGBM soft vote | 42 base | 150 trials each, SMOTE inside GroupKFold; submitted twice |
| submission_baseline_rf_20260518_185106_01.csv | 20260518_185106 | 55326130 | ? | RandomForest | 42 base | 200 trees, max_depth=15, class_weight=balanced |
| submission_lgb_acc_no_smote_20260519_053954_01.csv | 20260519_053954 | d12577a5 | 0.7515 | LightGBM | 42 base | accuracy-tuned; 60 trials; use_smote=False; worse than 0.7653 best |
| submission_lgb_acc_smote_20260519_055054_01.csv | 20260519_055054 | 5c0f5d5f | 0.7572 | LightGBM | 42 base | accuracy-tuned; 60 trials; use_smote=True; better than no-SMOTE today but worse than 0.7653 best |
| submission_xgb_acc_no_smote_20260519_060756_01.csv | 20260519_060756 | 7d02a4ed | ? | XGBoost | 42 base | accuracy-tuned; 60 trials; no SMOTE |
| submission_cnn_raw_sequence_20260519_060812_01.csv | 20260519_060812 | 3d6feafa | 0.6709 | 1D CNN | raw 300x6 sequence | epochs=6; validation from grouped split; not competitive |
