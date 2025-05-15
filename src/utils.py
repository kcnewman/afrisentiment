from src.preprocessing import Preprocess
import numpy as np


def build_freqs(texts, ys):
    """Build frequency tables
    Args:
        text: A list of str
        ys: An array matching sentiment
    Output:
        freqs (dict): Dictionary mapping each word(word, sentiment) pair to its frequency
    """
    preprocessor = Preprocess()
    yls = np.squeeze(ys).tolist()
    freqs = {}
    for y, text in zip(yls, texts):
        tokens = preprocessor.process(text)
        for word in tokens:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def train_naive_bayes(freqs, train_x, train_y):
    """Train a naive bayes classifier
    Args:
        freqs (dict): dictionary from (word,label) to how often the word appears
        train_x (ls): list of tweets
        train_y (ls): list of corresponding sentiment
    Output:
        logprior: the log prior
        loglikelihood: the loglikelihood
    """
    labels = np.array(train_y)
    classes = np.unique(labels)
    n_classes = len(classes)

    vocab = sorted(set([word for word, label in freqs.keys()]))
    vocab_size = len(vocab)

    word_idx = {word: i for i, word in enumerate(vocab)}
    class_idx = {c: i for i, c in enumerate(classes)}

    D = len(train_x)
    D_class = np.array([np.sum(labels == c) for c in classes])

    logprior = np.log(D_class / D)

    word_counts = np.zeros((n_classes, vocab_size), dtype=np.float64)

    for (word, label), count in freqs.items():
        i = class_idx[label]
        j = word_idx[word]
        word_counts[i, j] += count

    total_class_counts = word_counts.sum(axis=1, keepdims=True)
    alpha = 1.0
    loglikelihood = np.log(
        (word_counts + alpha) / (total_class_counts + alpha * vocab_size)
    )
    return logprior, loglikelihood, vocab, classes


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
