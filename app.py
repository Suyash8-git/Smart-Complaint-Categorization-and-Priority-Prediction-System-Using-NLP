import pandas as pd
import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Smart Complaint Analyzer", page_icon="📝", layout="centered")
st.title("📝 Smart Complaint Analyzer")
st.caption("NLP + Machine Learning based complaint categorization and priority prediction")

tab1, tab2 = st.tabs(["Analyze complaint", "Dashboard"])

with tab1:
    samples = {
        "— choose a sample —": "",
        "Wi-Fi down before exam": "The Wi-Fi in Lab 3 has stopped working and our practical exam is today.",
        "Projector issue": "The projector in classroom 204 is not turning on.",
        "No electricity": "There is no electricity in the computer laboratory.",
        "Security breach": "Someone entered the restricted server room without permission.",
    }
    pick = st.selectbox("Try a sample (optional)", list(samples))
    text = st.text_area("Enter your complaint:", value=samples[pick], height=120,
                        placeholder="e.g. The Wi-Fi in Lab 3 is not working...")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter a complaint first.")
        else:
            r = predict(text)
            c1, c2 = st.columns(2)
            c1.metric("Category", r["category"], f"{r['confidence']*100:.1f}% confidence")
            c2.metric("Priority", r["priority"], f"{r['priority_confidence']*100:.1f}% confidence")
            st.success(f"**Department:** {r['department']}")
            st.info(f"**Suggested action:** {r['action']}")
            st.subheader("Category probabilities")
            st.bar_chart(pd.Series(r["category_probs"]))

with tab2:
    df = pd.read_csv("data/complaints.csv")
    st.metric("Complaints in dataset", len(df))
    st.subheader("By category"); st.bar_chart(df["category"].value_counts())
    st.subheader("By priority")
    st.bar_chart(df["priority"].value_counts().reindex(["Low", "Medium", "High", "Critical"]))
    st.dataframe(df.head(50))
