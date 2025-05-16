import re
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
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


class DatasetPreprocessor:
    def __init__(self, input_dir, output_dir):
        """
        Args:
            input_dir (str): Directory containing raw CSV files.
            output_dir (str): Directory where preprocessed files will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_file(self, filename):
        """
        Loads, filters, encodes, and saves a single file.
        Args:
            filename (str): Name of the file (e.g., 'masakhane_afrisenti_twi_train.csv').
        """
        filepath = os.path.join(self.input_dir, filename)
        df = pd.read_csv(filepath)
        df = df[df["label"] != "neutral"]
        output_path = os.path.join(
            self.output_dir, filename.replace(".csv", "_preprocessed.csv")
        )
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    def preprocess_all(self):
        """
        Process all relevant files (train, validation, test) in the input directory.
        """
        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                self.preprocess_file(file)
