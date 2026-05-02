import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data with SEMICOLON separator (UCI/Kaggle format)
df = pd.read_csv("data/bank-full.csv", sep=';')

print("=" * 60)
print("📊 BANK MARKETING DATASET - LOAN ACCEPTANCE PREDICTION")
print("=" * 60)

# Data exploration
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nTarget Distribution:")
print(df['y'].value_counts())

# Drop duration (not known before call per UCI recommendation)
df = df.drop('duration', axis=1)

# Encode target
df['y_binary'] = (df['y'] == 'yes').astype(int)

# Features and target
X = df.drop(['y', 'y_binary'], axis=1)
y = df['y_binary']

# Identify column types (UPDATED for real UCI dataset)
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                        'contact', 'month', 'poutcome']  # Removed 'day_of_week', added 'poutcome'

numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']  # Added 'balance', 'day'

# One-hot encode
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
feature_columns = X_encoded.columns.tolist()

print(f"\nEncoded Features: {len(feature_columns)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Logistic Regression
print("\n📊 LOGISTIC REGRESSION")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lr_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, lr_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, lr_prob):.4f}")

# Decision Tree
print("\n🌳 DECISION TREE")
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=100, min_samples_leaf=50, 
                              random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, dt_pred):.4f}")
print(f"Precision: {precision_score(y_test, dt_pred):.4f}")
print(f"Recall:    {recall_score(y_test, dt_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, dt_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, dt_prob):.4f}")

# Feature importance
fi = pd.DataFrame({'feature': X_train.columns, 'importance': dt.feature_importances_})
print("\n🔝 TOP 10 FEATURES:")
print(fi.sort_values('importance', ascending=False).head(10).to_string(index=False))

# Save
joblib.dump(lr, "lr_model.pkl")
joblib.dump(dt, "dt_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "columns.pkl")
joblib.dump(numeric_features, "numeric_features.pkl")

print("\n✅ Models saved successfully!")