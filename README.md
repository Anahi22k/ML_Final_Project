# ML Final Project: Stroke Prediction

## Overview  
This project builds and evaluates machine-learning models to predict the risk of stroke based on demographic and health indicators. Using a publicly available stroke dataset, we explore the data, preprocess features, and compare several classification algorithms—Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and a Neural Network—addressing class imbalance with SMOTE.  

## Dataset  
- **Source**: `data/data.csv`  
- **Size**: 5,110 records × 12 features  
- **Features**  
  - `id` – Unique patient identifier  
  - `gender` – Male / Female  
  - `age` – Patient age in years  
  - `hypertension` – 0: No, 1: Yes  
  - `heart_disease` – 0: No, 1: Yes  
  - `ever_married` – Yes / No  
  - `work_type` – Private, Self-employed, Govt_job, children, Never_worked  
  - `Residence_type` – Urban / Rural  
  - `avg_glucose_level` – Average blood glucose level  
  - `bmi` – Body mass index (201 missing values imputed with median)  
  - `smoking_status` – formerly smoked, never smoked, smokes, Unknown (1,544 “Unknown” entries)  
  - `stroke` – Target: 0 = no stroke, 1 = stroke (4.87% positive cases)  

## Exploratory Data Analysis  
- **Missing Values**  
  - 201 missing BMI values (filled with the dataset median).  
  - No other missing entries after inspection.  
- **Class Imbalance**  
  - 4.87% of individuals experienced a stroke; the remaining 95.13% did not.  
- **Correlation Insights**  
  | Feature              | Correlation with Stroke |
  |----------------------|-------------------------:|
  | **age**              | 0.245                    |
  | **heart_disease**    | 0.135                    |
  | **avg_glucose_level**| 0.132                    |
  | **hypertension**     | 0.128                    |
  | ever_married         | 0.108                    |
  | bmi                  | 0.036                    |
  | gender, residence, work_type, smoking_status | near-zero or slightly negative  

## Data Preprocessing  
1. **Imputation**: BMI missing values → median  
2. **Encoding**:  
   - Binary: `gender`, `ever_married`, `Residence_type`  
   - Ordinal: `work_type` → 0–4  
   - Smoking: 0–3  
3. **Dropped**: `id`  
4. **Split**: 85% train / 15% test  
5. **SMOTE**: Balance stroke vs. non-stroke in training set  

## Modeling Approach  
- **Algorithms**:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Gradient Boosting  
  - MLP Neural Network  
- **Metrics**: Accuracy, Confusion Matrix, Precision, Recall, F1-Score  

## Results & Findings  
- **Logistic Regression**:  
  - Age and heart disease strongest predictors (odds ratios).  
- **Ensembles** (RF, XGBoost, GB): ≥97% accuracy, strong stroke-class recall.  
- **Neural Network**: ~95–97% accuracy.  
- **Best**: XGBoost / Random Forest for balanced performance.  

## File Structure  
. └── data/ ├── data.csv └── analysis.ipynb

## Requirements  
in bash:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

Usage
Clone the repo

cd data/

jupyter notebook analysis.ipynb


## Authors  
- Jason A, Kimberly, Oliver

## Acknowledgements  
- Dataset from [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- Inspired by tutorials from scikit-learn documentation  

## License  
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
