🚀 BankPulse Pro — Loan Acceptance Intelligence

An advanced Machine Learning + Luxury Analytics Dashboard for predicting customer acceptance of personal loan offers using the UCI Bank Marketing Dataset.

🧠 Project Overview

This project builds an end-to-end classification system that:

Predicts whether a customer will accept a loan offer
Provides real-time prediction via Streamlit dashboard
Visualizes customer segments, campaign performance, and insights
Compares Logistic Regression vs Decision Tree models

👉 Built as a portfolio-grade SaaS-style dashboard with enterprise-level UI.

🎯 Objective

Predict which customers are more likely to accept a personal loan offer based on:

Demographics (age, job, marital status)
Financial data (balance, loans)
Campaign interaction history
📂 Dataset
Source: UCI Machine Learning Repository
Dataset: Bank Marketing Dataset
File: bank-full.csv
⚙️ Tech Stack
Python 🐍
Pandas / NumPy
Scikit-learn
Streamlit (Frontend Dashboard)
Plotly (Interactive Visualizations)
Joblib (Model Serialization)
🧪 Models Used
🔹 Logistic Regression
Balanced class weights
Scaled numerical features
Strong baseline performance
🌳 Decision Tree Classifier
Controlled depth to prevent overfitting
Provides feature importance insights
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
ROC-AUC
🏗️ Project Structure
📁 project/
│
├── train.py              # Model training & evaluation
├── app.py                # Streamlit dashboard
├── data/
│   └── bank-full.csv     # Dataset
│
├── lr_model.pkl          # Logistic Regression model
├── dt_model.pkl          # Decision Tree model
├── scaler.pkl            # StandardScaler
├── columns.pkl           # Feature columns
├── numeric_features.pkl  # Numeric feature list
🚀 How to Run
1️⃣ Install dependencies
pip install pandas numpy scikit-learn streamlit plotly joblib
2️⃣ Train the models
python train.py
3️⃣ Run the dashboard
streamlit run app.py
🖥️ Dashboard Features

This project includes a premium SaaS-style dashboard:

📊 Executive Dashboard
KPI cards (Conversion Rate, Revenue Potential)
Customer segmentation charts
Campaign performance trends
🔮 Predictor Module
Real-time customer input
Loan acceptance probability
Risk analysis insights
📈 Analytics
Feature importance visualization
Age distribution & acceptance trends
Customer segmentation matrix
⚙️ Model Insights
Model comparison (LR vs Decision Tree)
Confusion matrix visualization
Business intelligence insights
💡 Key Business Insights
🎯 Students & retirees show highest acceptance rates
💰 Account balance is strongest predictor
📞 Too many contacts reduce conversion
📅 Campaign timing affects success
🔄 Previous success increases acceptance probability
🎨 UI Highlights
Luxury fintech-inspired design
Dark/Light theme switching
Glassmorphism & gradient UI
Interactive charts (Plotly)
Enterprise dashboard layout
📌 Example Output
Acceptance Probability: 72.4%
Decision: ACCEPTED / DECLINED
Risk Factors: Highlighted dynamically
📈 Skills Demonstrated
Machine Learning (Classification)
Feature Engineering
Model Evaluation
Data Visualization
Streamlit App Development
Business Insight Extraction
🔥 Future Improvements
Add XGBoost / LightGBM
Hyperparameter tuning (GridSearchCV)
Model deployment (FastAPI + Docker)
User authentication (SaaS version)
Real-time API integration
👨‍💻 Author

Mudassir Hassan
Data Science & AI Enthusiast

⭐ If You Like This Project

Give it a ⭐ on GitHub and share feedback!
