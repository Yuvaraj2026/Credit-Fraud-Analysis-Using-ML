This project compares three classifiers: Logistic Regression, XGBoost and LightGBM. Each is tuned with cross-validated grid search and evaluated with help of AUPRC. XGBoost performed best on the held-out test set.

Details about Dataset:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle / ULB Machine Learning Group)
1. 284,807 transactions made by European cardholders
2. 492 frauds (0.173% of all transactions)
3. Features V1-V28 are PCA-transformed (anonymised), plus Amount
4. No missing values

My approach:
1. Preprocessing: dropped 'Time' and separated the 'Class' target.
2. Split: 80/20 train/test split, stratified to preserve the fraud ratio.
3. Imbalance Handling: 'class_weight="balanced"' for Logistic Regression and 'scale_pos_weight' for XGBoost and LightGBM.
4. Scaling: StandardScaler inside a scikit-learn Pipeline for Logistic Regression, so scaling is fitted only on training folds (no data leakage).
5. Tuning: GridSearchCV with 5-fold StratifiedKFold, optimising Average Precision (AUPRC).
6. Evaluation: AUPRC on the held-out test set.

Why did we use AUPRC?

With 99.8% legitimate transactions, a model that always predicts "not fraud" scores 99.8% accuracy while catching no fraud. Average Precision (area under the precision-recall curve) measures how well a model finds the rare positive class.

Results:

| Model | CV AUPRC | Test AUPRC |

| XGBoost | 0.853 | 0.861 |
| LightGBM | 0.788 | 0.790 |
| Logistic Regression | 0.757 | 0.710 |

XGBoost achieved the highest AUPRC and outperformed the logistic regression baseline by 0.15.

Author:
Yuvaraj Mandal, www.linkedin.com/in/yuvaraj-mandal-995b45217, yuvarajmandal02@gmail.com


