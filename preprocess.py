import re

class TextPreprocessor:

    def normalize(self, text: str) -> str:

        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Normalize excessive punctuation
        text = re.sub(r"[!?.]{2,}", ".", text)

        # TODO (proposal alignment):
        # - leetspeak normalization (h4te -> hate)
        # - slang expansion
        # - subword-aware cleaning if needed

        return text.strip()