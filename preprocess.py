import re
import string

# =====================================================
# BASIC LEXICON (expandable)
# =====================================================

LEETSPEAK_MAP = {
    "h4te": "hate",
    "h@te": "hate",
    "l0ve": "love",
    "d1e": "die",
}

CONTRACTIONS = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "you're": "you are",
    "i'm": "i am",
}


# =====================================================
# PREPROCESSOR CLASS
# =====================================================

class TextPreprocessor:

    def __init__(self):
        pass

    # -----------------------------
    # Main pipeline
    # -----------------------------
    def normalize(self, text: str) -> str:

        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = self.remove_urls(text)
        text = self.remove_html(text)
        text = self.expand_contractions(text)
        text = self.normalize_leetspeak(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)

        return text.strip()

    # -----------------------------
    # URL removal
    # -----------------------------
    def remove_urls(self, text):
        return re.sub(r"http\S+|www\S+", "", text)

    # -----------------------------
    # HTML cleanup
    # -----------------------------
    def remove_html(self, text):
        return re.sub(r"<.*?>", "", text)

    # -----------------------------
    # Contractions expansion
    # -----------------------------
    def expand_contractions(self, text):

        for k, v in CONTRACTIONS.items():
            text = text.replace(k, v)

        return text

    # -----------------------------
    # Leetspeak normalization
    # -----------------------------
    def normalize_leetspeak(self, text):

        for k, v in LEETSPEAK_MAP.items():
            text = text.replace(k, v)

        return text

    # -----------------------------
    # Remove punctuation
    # -----------------------------
    def remove_punctuation(self, text):

        return text.translate(
            str.maketrans("", "", string.punctuation)
        )

    # -----------------------------
    # Normalize whitespace
    # -----------------------------
    def normalize_whitespace(self, text):

        return re.sub(r"\s+", " ", text)