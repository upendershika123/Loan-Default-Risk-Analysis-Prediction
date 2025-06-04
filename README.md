# 📉 Loan Default Risk Analysis & Prediction

This project is a comprehensive loan default risk analysis and prediction system built using Python and machine learning techniques. It helps users understand key factors influencing loan defaults through Exploratory Data Analysis (EDA) and then predicts the risk of loan default using a trained classification model.

---

## 📌 Project Objectives

- 📊 Perform Exploratory Data Analysis to uncover patterns and relationships in loan applicant features  
- 🧼 Preprocess the dataset to handle missing values and categorical features  
- 🤖 Train classification models to predict loan default risk  
- 📈 Evaluate model performance with accuracy, precision, recall, F1-score, and ROC-AUC  
- 💡 Provide insights into important features driving loan defaults  

---

## 🗂️ Project Directory Structure

Loan-Default-Risk-Analysis-Prediction/
├── data/
│ └── loan_data.csv # Original dataset used for analysis
├── notebooks/
│ └── loan_default_EDA.ipynb # EDA and visualization notebook
├── src/
│ ├── data_preprocessing.py # Data cleaning and feature engineering
│ ├── model_training.py # Model training and evaluation
│ └── prediction.py # Prediction on new/unseen data
├── requirements.txt # Required Python packages
├── README.md # Project documentation (this file)
└── .gitignore # Files and folders to exclude from Git tracking

yaml

---

## 📊 Exploratory Data Analysis (EDA)

The `notebooks/loan_default_EDA.ipynb` performs:

- Distribution analysis of the target variable `Loan Default`  
- Visualization of key features vs default rate (e.g., Income, Credit Score)  
- Correlation heatmaps of numerical variables  
- Boxplots and countplots for categorical variables  
- Identification of missing values and data quality issues  

🖼️ All graphs and insights are included in the notebook for easy review.

---

## 🧼 Data Preprocessing

The `src/data_preprocessing.py` script handles:

- Handling missing values (mean imputation for numeric, mode or 'Unknown' for categoricals)  
- Encoding categorical variables using One-Hot Encoding or Label Encoding  
- Feature scaling (if applicable)  
- Splitting data into features (`X`) and target (`y`)

---

## 🤖 Model Training and Prediction

The `src/model_training.py` script performs:

- Splitting data into training and test sets  
- Training classification models such as Logistic Regression, Random Forest, or XGBoost  
- Evaluating models with classification metrics and ROC-AUC  
- Saving the best model for future prediction  

The `src/prediction.py` script demonstrates how to use the trained model to predict loan default risk on new applicant data.

---

## 💻 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/upendershika123/Loan-Default-Risk-Analysis-Prediction.git
   cd Loan-Default-Risk-Analysis-Prediction
(Optional) Create and activate a virtual environment

bash
python -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Run Jupyter notebook for EDA and experimentation

bash
jupyter notebook notebooks/loan_default_EDA.ipynb
Run scripts for preprocessing, training, and prediction

📥 Dataset Information
Dataset: Loan application records including applicant details, financial data, and loan performance status
Source: (Mention source if applicable)
Format: CSV file with numeric and categorical columns describing loan applicants and loan outcome

💡 Features Used in the Model (Examples)
Feature	Description
Age	Applicant’s age
Income	Applicant’s monthly/annual income
CreditScore	Credit score of applicant
EmploymentType	Type of employment (salaried, self-employed)
LoanAmount	Amount of loan requested
LoanTerm	Duration of loan term
HasCoSigner	Whether loan has co-signer (Yes/No)
Default	Target variable (1 if defaulted, 0 otherwise)

📈 Sample Output
mathematica

Loan Default Risk Prediction for Applicant:

Predicted Risk: HIGH
Probability of Default: 82%

Key factors influencing prediction:
- Low Credit Score
- High Loan Amount
- Short Employment Duration
📦 requirements.txt
nginx

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
jupyter
🧾 .gitignore
markdown

__pycache__/
*.pyc
*.pyo
*.pyd
.env
.ipynb_checkpoints/
.DS_Store
🤝 Contribution Guidelines
Contributions are welcome! If you want to:

Improve model accuracy or try new algorithms

Add new EDA visualizations

Enhance data preprocessing and feature engineering

Improve documentation or add examples

Please submit a Pull Request or open an Issue to discuss your ideas.

🧑‍💻 Author
Upender Shika
📧 (upendershika206@gmail.com)
🔗 https://github.com/upendershika123

css
If you want, I can also help generate a **requirements.txt** file or a sample script to get you started!
