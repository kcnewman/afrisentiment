import re
from nltk.tokenize import TweetTokenizer


class Preprocess:
    def __init__(self, stop_words = None):
        self.tokenizer = TweetTokenizer()
        self.stop_words = stop_words if stop_words else set()
        
    def clean_text(text):
        """
        Input:
            text: a string containing corpus
        Output:
            text_clean: clean text(lowercase, punctuation, tags, etc)
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]','',text)
        text = re.sub(r'^RT[\s]+','',text)
        text = re.sub(r'https?://[^\s\n\r]+', '', text)
        text = re.sub(r'#', '', text)
        

    def tokenize(text):
        "Split cleaned text into tokens."
        pass
    
