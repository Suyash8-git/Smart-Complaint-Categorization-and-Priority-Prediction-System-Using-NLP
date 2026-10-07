"""Prediction helper used by the Streamlit app and for quick CLI tests."""
import joblib
from src.preprocessing import tokenize  # noqa: F401  (needed to unpickle the vectorizer)

DEPARTMENT = {
    "Internet/Wi-Fi": "IT Support", "Hardware": "Technical Support", "Software": "IT Support",
    "Electricity": "Electrical Department", "Maintenance": "Maintenance Department",
    "Security": "Security Department", "Other": "General Helpdesk",
}
ACTION = {
    "Internet/Wi-Fi": "Check router/network connection and contact IT Support.",
    "Hardware": "Inspect the device and assign Technical Support.",
    "Software": "Check application installation/configuration.",
    "Electricity": "Contact Electrical Department immediately.",
    "Maintenance": "Forward complaint to Maintenance Department.",
    "Security": "Immediately notify Security Department.",
    "Other": "Forward to General Helpdesk.",
}
_cat = _pri = None


def _load():
    global _cat, _pri
    if _cat is None:
        _cat = joblib.load("models/category_model.pkl")
        _pri = joblib.load("models/priority_model.pkl")


def predict(text: str) -> dict:
    _load()
    probs = _cat.predict_proba([text])[0]
    cat = _cat.classes_[probs.argmax()]
    ppri = _pri.predict_proba([text])[0]
    return {
        "category": cat,
        "confidence": float(probs.max()),
        "category_probs": dict(sorted(zip(_cat.classes_, probs), key=lambda x: -x[1])),
        "priority": _pri.classes_[ppri.argmax()],
        "priority_confidence": float(ppri.max()),
        "department": DEPARTMENT[cat],
        "action": ACTION[cat],
    }


if __name__ == "__main__":
    import sys
    r = predict(" ".join(sys.argv[1:]) or "The Wi-Fi in Lab 3 has stopped working and our practical exam is today.")
    for k, v in r.items():
        print(f"{k:20}: {v}")
