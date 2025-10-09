The project aims to detect fraudulent transactions in a highly imbalanced credit card dataset using machine learning models. The workflow includes data preprocessing, handling imbalance, model training, hyperparameter tuning, and performance evaluation.

The link of the kaggle dataset is: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

1. Exploratory Data Analysis:

No missing values were found. We discovered correlation between features using a Pearson heatmap. Feature distributions were analyzed using KDE plots to compare fraudulent and non-fraudulent transactions.

2. Data Preprocessing:

We dropped the Time column as it isn't really predictive for fraudulent transcactions. We Split the data into training and testing sets (80:20) with stratification to preserve the class ratio. Then we standarised the Amount feature using StandardScaler (fitting on the training dataset). The rest of th variables were PCA transformed and hence, were already standardised. 

3. Handling Imbalanced Data

Since the dataset is extremely imbalanced, with fraud cases representing less than 0.2% of total transactions, to address this, the SMOTE (Synthetic Minority Oversampling Technique) method was applied on the training data to create a balanced dataset for logistic regression that does not natively handle class imbalance.

4. Model Building

Three machine learning algorithms were implemented:

Logistic Regression with L1 and L2 loss functions, 
XGBoost Classifier and LightGBM Classifier where imbalance was handled using scale_pos_weight.

Each model was trained using GridSearchCV for hyperparameter optimization and 3-fold cross-validation.
Since we are using GridSearch CV, the choices of hyperparameters were chosen based on commonly used values and tuned to maximize the Average Precision (AUPRC) score.



