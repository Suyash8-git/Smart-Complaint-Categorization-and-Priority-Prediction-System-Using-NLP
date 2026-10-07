"""Text preprocessing for the complaint classifier."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations are KEPT: "not working" must stay different from "working".
KEEP = {"not", "no", "nor", "cannot", "never", "none"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP


def clean_text(text: str) -> str:
    """Lowercase, normalise Wi-Fi spellings/contractions, strip symbols."""
    text = str(text).lower()
    text = re.sub(r"wi[\s-]?fi", "wifi", text)
    text = text.replace("can't", "cannot")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    """Clean -> split into tokens -> remove stopwords."""
    return [w for w in clean_text(text).split() if w not in STOPWORDS]


if __name__ == "__main__":
    s = "The Wi-Fi in LAB 3 is NOT working!!!"
    print(clean_text(s))
    print(tokenize(s))
