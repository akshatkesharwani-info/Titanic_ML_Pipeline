# Titanic Survival Prediction - Complete ML Pipeline

An end-to-end machine learning project on the Titanic dataset: data exploration, feature engineering, training 3 models with cross-validation, evaluation, hyperparameter tuning and final predictions for Kaggle.

## Objective

Predict whether a passenger survived the Titanic and compare Logistic Regression, Random Forest and XGBoost to find the best model.

## Dataset

- `train.csv`: 891 passengers with the target column `Survived`
- `test.csv`: 418 passengers without the target
- Columns: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

## Project Structure

```
Titanic_ML_Pipeline/
|-- Titanic_ML_Pipeline.ipynb   # full pipeline notebook
|-- submission.csv              # predictions for test.csv (PassengerId, Survived)
|-- train.csv
|-- test.csv
|-- requirements.txt
|-- README.md
|-- images/                     # all charts saved as PNG (11 files)
```

## How to Run

1. Unzip the project folder.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```
   jupyter notebook Titanic_ML_Pipeline.ipynb
   ```
4. Click **Run All**. This creates `submission.csv` and the charts in `images/`.

**Google Colab:** upload `train.csv` and `test.csv` in the Files panel first, then upload and run the notebook. Keep both CSV files in the same folder as the notebook.

## Pipeline

**1. Data Exploration**
- Feature types, distributions and missing values (Age 19.87%, Cabin 77.10%, Embarked 0.22%)
- Survival rate by Sex, Pclass, Embarked and Age group
- Correlation of every feature with survival

**2. Feature Engineering**
- Age: filled with the median Age of each Pclass
- Embarked: filled with the mode (S)
- Cabin: converted to a `Has_Cabin` feature
- New features: `FamilySize`, `IsAlone`, `Title` (from Name), `AgeBin`
- Encoding: Sex as 0/1, one-hot encoding for Embarked, Title and AgeBin
- Scaling: StandardScaler on Age, Fare and FamilySize (fitted on training data only)

**3. Model Training**
- Stratified 80/20 train and validation split
- Logistic Regression, Random Forest, XGBoost
- 5-fold stratified cross-validation for every model

**4. Model Evaluation**
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix for each model, ROC curve comparison, feature importance plots
- Best model selected by F1-Score

**5. Hyperparameter Tuning**
- GridSearchCV (5-fold, scoring = F1) on the best model with 4 hyperparameters
- Before and after comparison, plus an extra tuning check on Random Forest and XGBoost

**6. Final Prediction**
- Tuned best model trained again on the full train data and used to predict `test.csv`

## Model Comparison

Results on the 20% validation set (179 passengers):

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8380** | **0.8030** | **0.7681** | **0.7852** | **0.8762** | **0.7779** |
| Random Forest | 0.8045 | 0.7429 | 0.7536 | 0.7482 | 0.8250 | 0.7527 |
| XGBoost | 0.8045 | 0.7429 | 0.7536 | 0.7482 | 0.8323 | 0.7543 |

**Best model: Logistic Regression** (highest F1-Score).

## Hyperparameter Tuning

Logistic Regression was tuned with GridSearchCV over `C`, `penalty`, `solver` and `class_weight`.

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 0.8380 | 0.8380 |
| Precision | 0.8030 | 0.8030 |
| Recall | 0.7681 | 0.7681 |
| F1-Score | 0.7852 | 0.7852 |
| AUC-ROC | 0.8762 | 0.8762 |

Best parameters: `C=1`, `penalty=l2`, `solver=saga`, `class_weight=None`. This is nearly the same as the default model, so the score did not change.

As an extra check, Random Forest and XGBoost were also tuned. They did not beat Logistic Regression:

| Model | F1 Before | F1 After | AUC-ROC After |
|---|---|---|---|
| Random Forest | 0.7482 | 0.7328 | 0.8514 |
| XGBoost | 0.7482 | 0.7313 | 0.8370 |

## Key Insights

- Women survived much more than men (74.2% vs 18.9%)
- 1st class survival was 63.0% and 3rd class only 24.2%
- Children had 58.0% survival and seniors only 22.7%
- `Title`, `Has_Cabin` and `IsAlone` (new features) all have a clear relation with survival
- Most important features: Sex, Title, Pclass, Has_Cabin and Fare
- On this small dataset, the simple Logistic Regression was better than the tree models

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn, xgboost
- Jupyter Notebook

## Author

**Akshat Kesharwani**
- Portfolio: https://akshatkesharwani-info.github.io/
- GitHub: https://github.com/akshatkesharwani-info
