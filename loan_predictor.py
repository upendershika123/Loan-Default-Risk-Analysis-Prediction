
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

df = pd.read_csv("Loan_default.csv")  

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

y = df['Default']

features = ['Income', 'LoanAmount', 'CreditScore', 'DTIRatio']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print(f"AUC Score: {auc:.3f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Histogram – Loan Amount Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['LoanAmount'], kde=True, color='skyblue')
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Bar Plot – Average Loan Amount by DTI Ratio Range
df['DTIRange'] = pd.cut(df['DTIRatio'], bins=5)
plt.figure(figsize=(6, 4))
sns.barplot(x='DTIRange', y='LoanAmount', data=df)
plt.title('Loan Amount vs DTI Ratio')
plt.xlabel('DTI Ratio Range')
plt.ylabel('Average Loan Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Pie Chart – Distribution of Loan Purposes
plt.figure(figsize=(5, 5))
df['LoanPurpose'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Purposes')
plt.ylabel('')
plt.show()

# Box Plot – Loan Amount by Credit Score Range
df['CreditScoreRange'] = pd.cut(df['CreditScore'], bins=4)
plt.figure(figsize=(6, 4))
sns.boxplot(x='CreditScoreRange', y='LoanAmount', data=df)
plt.title('Loan Amount by Credit Score')
plt.xlabel('Credit Score Range')
plt.ylabel('Loan Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
