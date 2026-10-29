## Deployment Link : https://ardentix-round-2-harsha.streamlit.app/

# Ardentix Round-2
Skill-to-Role Mapping ML Project
 ##  Ardentix Round-2
Skill-to-Role Mapping with Actionable Career Tips using Machine Learning
## 📌 Project Overview

Many learners acquire technical skills but are unaware of the full range of job roles those skills already qualify them for.
They often focus on one popular role and miss adjacent or hidden roles that require almost the same skill set.
This project uses Machine Learning on real job data to:

Map a user’s skills to multiple suitable job roles

Provide actionable skill tips showing what additional skills can unlock new opportunities

The system is simple, explainable, ethical, and hackathon-safe.

## 🎯 Problem Statement

Learners lack data-driven clarity about:

Which job roles they are already eligible for

What small skill upgrades can expand their career options

Most existing career guidance is:

Generic

Opinion-based

Focused on only popular roles

This project solves that using historical job skill–role patterns.

## 🧠 Solution Summary

The system:
Takes user skills as text input
Uses a trained ML model to:
Predict Top-N eligible job roles
Compute confidence scores
Analyzes learned model weights to:
Identify important skills for top roles
Recommend missing skills as actionable tips
No aptitude tests.
No personality analysis.
No future prediction.
Just clear, data-driven guidance.

<img width="884" height="556" alt="image" src="https://github.com/user-attachments/assets/1f7b4319-0a6f-4810-971c-9d2786894350" />



##   📊 Dataset Description

Source: Real job descriptions dataset (downloaded manually, not via API)

Key columns used:

skills – Required job skills (text)

Role – Job role/title (target variable)

⚠️ No synthetic data is used.
⚠️ Dataset is preprocessed and stored locally.

## 🔄 Complete ML Workflow (Step-by-Step)
1️⃣ Data Loading

Loaded real job dataset

Selected only relevant columns (skills, Role)

Removed missing values

2️⃣ Problem Framing

Task type: Multi-class classification

Input: Textual skills

Output: Job role category

3️⃣ Data Preparation
Text Cleaning

Lowercasing

Removal of noise & irrelevant tokens

Basic normalization

Feature Engineering (TF-IDF)

Converted skills text into numerical vectors

Used:

ngram_range=(1,2)

min_df=3

max_features=5000

L2 normalization

⚠️ TF-IDF was fitted only on training data to prevent data leakage.

4️⃣ Train–Test Split

Used Stratified 80–20 split

Reason:

Dataset is imbalanced

Stratification preserves role distribution

Ensures fair evaluation and realistic performance

5️⃣ Model Selection

Multiple models were explored conceptually:

Naive Bayes

Linear models

SGD-based classifiers

Final Choice: Logistic Regression

Fast and scalable for large text data

Probabilistic output (predict_proba)

Highly interpretable (feature weights → skill tips)

Industry-trusted baseline for NLP tasks

6️⃣ Model Training

Trained on a representative subset (~150k samples) for efficiency

Hyperparameters fixed for stability:

penalty = l2

C = 5

solver = lbfgs
max_iter = 1000
This ensures:
Fast training
No overfitting
Hackathon-time feasibility
7️⃣ Model Persistence
Saved artifacts using joblib:
final_logistic_model.joblib
tfidf_vectorizer.joblib
These files are the only dependencies used in deployment.
🌐 Deployment Architecture
Backend Logic
Load trained model and TF-IDF vectorizer
Transform user input
Predict top job roles
Extract important skills using model coefficients
Compare with user skills to suggest missing skills
Frontend (Streamlit)
Skill input box
Predict button with loading animation
Result cards for Top-5 roles
Dedicated skill-tip box
Beginner-friendly guidance for weak inputs


## 🖥️ How to Run Locally
1️⃣ Create Virtual Environment
python -m venv demo
demo\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Application
streamlit run app.py

## 🏁 Conclusion

This project demonstrates a complete, real-world ML pipeline:
Clear problem
Real data
Proper preprocessing
Interpretable modeling
Clean deployment
Immediate user value

