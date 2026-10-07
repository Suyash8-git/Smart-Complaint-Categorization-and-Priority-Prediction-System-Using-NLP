<<<<<<< HEAD
# Smart Complaint Categorization and Priority Prediction System (NLP)

## Run
    pip install -r requirements.txt
    python -m src.generate_dataset   # creates data/complaints.csv (skip if you built your own)
    python -m src.train              # trains, evaluates, saves models/ and reports/
    python -m src.predict "Wi-Fi is down in Lab 2"   # quick CLI test
    streamlit run app.py             # web UI

## Pipeline
Complaint -> clean/lowercase/tokenize/stopwords (negations kept) -> TF-IDF (1-2 grams) -> Logistic Regression / Naive Bayes
-> Category + Priority -> Department & Action (rule-based mapping)
=======
# Smart-Complaint-Categorization-and-Priority-Prediction-System-Using-NLP
>>>>>>> a7bb46da9c4400140b36d1f6569059125c5e5524
