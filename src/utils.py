from src.preprocessing import Preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_freqs(texts, ys):
    """Build frequency tables
    Args:
        text: A list of str
        ys: A binary array matching sentiment (1 for positive and 0 negative)
    Output:
        freqs (dict): Dictionary mapping each word(word, sentiment) pair to its frequency
    """
    yls = np.squeeze(ys).tolist()
    freqs = {}
    for y, text in zip(yls, texts):
        for word in Preprocess.process(text):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def plot_sentiment(df):
    sentiment_counts = df["label"].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Distribution of Sentiments")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.show()


def count_word_freqs_by_class(texts, labels):
    "return class-wise word freqs dict"
    pass


def compute_tf_idf(corpus):
    "compute TF-IDF vectors for a list of texts"


def compute_log_prior(labels):
    "compute log prior prob for each class"
    pass


def compute_log_likelihoods(freqs_by_class, vocab_size, alpha=1):
    "With Laplacian smoothing"
    pass


def predict_naive_bayes(text, log_priors, log_likelihoods, vocab, stopwords_set):
    "predict class for a given text"
    pass


def evaluate_predictions(y_true, y_pred):
    "Compute accuracy, precision, recall, F1."
    pass


def plot_confusion_matrix(y_true, y_pred, labels):
    "Plot a labeled confusion matrix."
    pass


def visualize_clusters(X_reduced, labels, texts):
    "PCA/t-SNE visualization of clusters with optional sentiment labels."
    pass


def cosine_similarity_matrix(query_vec, doc_vecs):
    "Return similarity scores."


def search_documents(query, docs, vectorizer):
    "Compute and return top-N similar documents."


def load_aligned_sentences(filepath):
    "Load Ewe–Twi parallel data."


def compute_avg_fasttext_embeddings(sentences, model):
    "Generate sentence embeddings."


def train_translation_matrix(X_src, Y_tgt):
    "Use least squares to learn transformation matrix."


def translate_sentence(sentence, src_model, W):
    "Project sentence to target embedding space."


def classify_translated_embedding(embedding, classifier):
    "Use trained model to predict sentiment."
