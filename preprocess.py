import re

class TextPreprocessor:

    def normalize(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^\w\s]", " ", text)

        # TODO:
        # - leetspeak normalization
        # - slang normalization
        # - adversarial text handling

        return text.strip()