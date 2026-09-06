# Customer Churn Prediction – Telco Dataset

## Project Overview

This project is an end-to-end Machine Learning system developed as part of Devixo Solutions Task 03.

It predicts the likelihood of a telecom customer churning based on their account, billing, and service usage data using multiple machine learning models, and presents the results through a web-based prediction interface built with Streamlit.

---

## Objective

- Build and compare multiple machine learning models
- Perform data preprocessing and feature engineering
- Evaluate models using appropriate metrics
- Tune hyperparameters and select the best-performing model
- Deploy the model using a Streamlit-based interface

---

## Technologies Used

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib
- Pickle

### Frontend / Deployment

- Streamlit

---

## Dataset

- Name: Telco Customer Churn Dataset
- File: Telco_Customer_Charm.csv
- Records: 7,043
- Type: Classification

---

## Project Structure

```
Customer_Churn/
│
├── dataset/
│   └── Telco_Customer_Churn.csv
│
├── notebook/
│   └── Task_03_ML.ipynb
│
├── report/
│   └── Task_03_ML.pdf
│
├── app.py
├── best_churn_model.joblib
├── scaler.joblib
├── model_columns.joblib
└── README.md
```

---

## Machine Learning Workflow

### Part 1 – Data Preprocessing

- Loaded dataset using Pandas
- Handled missing values (converted `TotalCharges` to numeric)
- Removed duplicates
- Engineered new features: `TotalServices`, `AvgMonthlySpend`
- Encoded categorical features using one-hot encoding
- Scaled numerical features using StandardScaler
- Split dataset into training and testing sets

---

### Part 2 – Model Development

The following models were trained:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

---

### Part 3 – Model Evaluation

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### Part 4 – Model Comparison

| Model             | Accuracy | Advantages                     | Limitations                       |
| ----------------- | -------- | ----------------------------- | ---------------------------------- |
| Decision Tree     | Moderate | Simple and interpretable      | Prone to overfitting               |
| Random Forest     | High     | Robust and accurate           | Higher computational cost          |
| SVM               | Moderate | Strong recall on churn class  | Sensitive to parameter choice      |
| Gradient Boosting | Moderate | Best recall, catches churners | Slower to train, more false alarms |

---

### Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to optimize model parameters. The best model was selected using:

```
best_model = grid.best_estimator_
```

---

### Part 5 – Conclusion

The tuned Random Forest model performed the best overall due to:

- Highest accuracy and precision on the test set
- Strong, stable F1 score across cross-validation folds
- Balanced trade-off between catching churners and avoiding false alarms

This model was selected and deployed in the Streamlit application.

---

## Model Deployment

The trained model, scaler, and feature columns were saved using:

```python
joblib.dump(best_model, "best_churn_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X_train.columns), "model_columns.joblib")
```

---

## Web Application (Dashboard)

### Features

- Interactive prediction interface
- Input of customer demographic, account, billing, and service details
- Real-time prediction
- Churn risk classification with probability score

---

## How to Run the Project

### 1. Install Dependencies

```
pip install streamlit joblib pandas scikit-learn
```

---

### 2. Run the Streamlit Application

```
streamlit run app.py
```

---

### 3. Open in Browser

http://localhost:8501

---

## Input Features

The model uses the following customer attributes:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

---

## Output

The system provides:

- Prediction (Churn / No Churn)
- Churn probability score

---

## Key Highlights

- Complete machine learning pipeline
- Four model comparison (Decision Tree, Random Forest, SVM, Gradient Boosting)
- Hyperparameter tuning and cross-validation
- Feature importance analysis
- Deployment using Streamlit
- Interactive prediction interface
- Real-world dataset

---

## Future Improvements

- Add SHAP-based explainability for individual predictions
- Integrate charts and analytics dashboard
- Handle class imbalance with SMOTE
- Deploy on cloud platforms

---

## Author

Maryam Khan
AI/ML Intern, Devixo Solutions

---