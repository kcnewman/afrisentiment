import re
from nltk.tokenize import TweetTokenizer


class Preprocess:
    def __init__(self, stop_words=None):
        self.tokenizer = TweetTokenizer(
            preserve_case=False, reduce_len=True, strip_handles=True
        )
        self.stop_words = stop_words if stop_words else set()

    def clean_text(self, text):
        """
        Input:
            text: a string containing corpus
        Output:
            text_clean: clean text(lowercase, punctuation, tags, etc)
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"^RT[\s]+", "", text)
        text = re.sub(r"https?://[^\s\n\r]+", "", text)
        text = re.sub(r"#", "", text)
        return text

    def tokenize(self, text):
        "Split cleaned text into tokens."
        return self.tokenizer.tokenize(self.clean_text(text))

    def remove_stopwords(self, tokens):
        return [tok for tok in tokens if tok not in self.stop_words]

    def process(self, text, remove_stopwords=False):
        tokens = self.tokenize(text)
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        return tokens
