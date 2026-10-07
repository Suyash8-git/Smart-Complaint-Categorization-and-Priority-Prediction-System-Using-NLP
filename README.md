# Smart Complaint Categorization and Priority Prediction System Using NLP

An NLP and Machine Learning based system that automatically analyzes complaints and predicts their **category, priority, department, and recommended action**.

## 🚀 Features

* Complaint text preprocessing
* TF-IDF feature extraction
* Category prediction
* Priority prediction
* Department mapping
* Recommended action
* Streamlit web application

## 🧠 Pipeline

```text
Complaint
   ↓
NLP Preprocessing
   ↓
TF-IDF
   ↓
ML Classification
   ↓
Category + Priority
   ↓
Department + Action
```

## 🛠️ Technologies

* Python
* NLP
* Scikit-learn
* Pandas
* TF-IDF
* Logistic Regression / Naive Bayes
* Streamlit

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python -m src.generate_dataset
python -m src.train
streamlit run app.py
```

## 📁 Project Structure

```text
data/       → Dataset
models/     → Trained models
reports/    → Evaluation results
src/        → NLP and ML code
app.py      → Streamlit application
```

