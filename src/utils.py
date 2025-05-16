import sys

sys.path.append("../")

from src.preprocessing import Preprocess
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    train_df = pd.read_csv(
        "../data/preprocessed/masakhane_afrisenti_twi_train_preprocessed.csv"
    )
    val_df = pd.read_csv(
        "../data/preprocessed/masakhane_afrisenti_twi_validation_preprocessed.csv"
    )
    test_df = pd.read_csv(
        "../data/preprocessed/masakhane_afrisenti_twi_test_preprocessed.csv"
    )

    encoder = LabelEncoder()
    train_df["sentiment"] = encoder.fit_transform(train_df["label"])
    val_df["sentiment"] = encoder.transform(val_df["label"])
    test_df["sentiment"] = encoder.transform(test_df["label"])

    return train_df, val_df, test_df, encoder


def build_freqs(texts, ys):
    """Build frequency tables
    Args:
        text: A list of str
        ys: An array matching sentiment
    Output:
        freqs (dict): Dictionary mapping each word(word, sentiment) pair to its frequency
    """
    preprocessor = Preprocess()
    yls = list(np.squeeze(ys).tolist()) if not isinstance(ys, list) else ys
    freqs = {}
    for y, text in zip(yls, texts):
        tokens = preprocessor.process(text)
        for word in tokens:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def extract_features(train_x, freqs):
    """
    Extract features from an array of texts.

    Args:
        train_x (list or np.array): a list or array of texts (tweets).
        freqs (dict): a dictionary mapping (word, sentiment) pairs to their frequency counts.

    Returns:
        X (np.ndarray): a 2D numpy array with shape (number_of_texts, vocabulary_size)
    """
    preprocessor = Preprocess()
    tokens_l = [preprocessor.process(tweet) for tweet in train_x]
    vocab = sorted(set([word for word, _ in freqs.keys()]))
    word_idx = {word: i for i, word in enumerate(vocab)}
    n_samples = len(tokens_l)
    n_features = len(vocab)
    X = np.zeros((n_samples, n_features))
    for i, tokens in enumerate(tokens_l):
        for word in tokens:
            if word in word_idx:
                X[i, word_idx[word]] += 1
    return X


def train_naive_bayes(freqs, train_x, train_y, alpha=1.0):
    """Train a naive bayes classifier
    Args:
        freqs (dict): dictionary from (word,label) to how often the word appears
        train_x (ls): list of tweets
        train_y (np.array): list of corresponding sentiment
        alpha (float): Laplacian smoothing value
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
    loglikelihood = np.log(
        (word_counts + alpha) / (total_class_counts + alpha * vocab_size)
    )
    return logprior, loglikelihood, vocab, classes


def predict_naive_bayes(
    text, logprior, loglikelihood, vocab, classes, fallback="prior", return_label="name"
):
    class_names = {0: "negative", 1: "positive"}
    preprocessor = Preprocess()
    tokens = preprocessor.process(text)
    word_idx = {word: i for i, word in enumerate(vocab)}
    class_scores = logprior.copy()

    found_word = False
    for word in tokens:
        if word in word_idx:
            idx = word_idx[word]
            class_scores += loglikelihood[:, idx]
            found_word = True
    if not found_word:
        if fallback == "prior":
            pred_class = classes[np.argmax(logprior)]
        else:
            return "unknown"
    else:
        pred_class = classes[np.argmax(class_scores)]

    if return_label == "name":
        return class_names[pred_class]
    else:
        return pred_class


def cross_validation(train_x, train_y, val_x, val_y, alphas, k=5):
    """
    Perform k-fold CV to tune alpha for Naive Bayes
    Args:
        train_x: list or array of texts (training data)
        train_y: array of encoded labels (training labels)
        val_x: list or array of texts (validation data)
        val_y: array of encoded labels (validation labels)
        alphas: list of alpha values to try
        k: number of folds
    Returns:
        best_alpha: alpha with highest average F1
        scores_dict: {alpha: avg_f1}
    """
    ult_alpha = None
    ult_score = -1
    scores_dict = {}

    for alpha in alphas:
        f1_scores = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(val_x):
            val_fold_x = val_x.iloc[val_idx]
            val_fold_y = val_y.iloc[val_idx]

            freqs = build_freqs(train_x, train_y)

            logprior, loglikelihood, vocab, classes = train_naive_bayes(
                freqs, train_x, train_y, alpha
            )

            preds = [
                predict_naive_bayes(
                    text,
                    logprior,
                    loglikelihood,
                    vocab,
                    classes,
                    return_label="encoded",
                )
                for text in val_fold_x
            ]

            f1 = f1_score(val_fold_y, np.array(preds), average="macro")
            f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores)
        scores_dict[alpha] = avg_f1

        if avg_f1 > ult_score:
            ult_score = avg_f1
            ult_alpha = alpha

    return ult_alpha, scores_dict


def retrain_final_model(full_train_x, full_train_y, alpha):
    freqs_full = build_freqs(full_train_x, full_train_y)
    return train_naive_bayes(freqs_full, full_train_x, full_train_y, alpha)


def evaluate_model(test_x, test_y, logprior, loglikelihood, vocab, classes, encoder):
    test_preds = [
        predict_naive_bayes(text, logprior, loglikelihood, vocab, classes)
        for text in test_x
    ]
    test_preds_enc = encoder.transform(test_preds)
    f1 = f1_score(test_y, test_preds_enc, average="macro")
    acc = accuracy_score(test_y, test_preds_enc)

    print(f"\nFinal Evaluation:\nF1 Score: {f1:.4f}\nAccuracy: {acc:.4f}")
    cm = confusion_matrix(test_y, test_preds_enc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap="Blues")


def print_sample_prediction(text, logprior, loglikelihood, vocab, classes):
    pred = predict_naive_bayes(text, logprior, loglikelihood, vocab, classes)
    print(f"\nSample prediction for '{text}': {pred}")


def build_tfidf(alpha=1.0, ngram_range=(1, 2), max_df=0.95, min_df=2):
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
        ),
        MultinomialNB(alpha=alpha),
    )


def tune_params(train_x, train_y, val_x, val_y, params):
    param_names = list(params.keys())
    param_values = list(params.values())
    best_score = 0
    best_params = None
    all_results = []

    print("Manual parameter search...\n")

    for param_ in itertools.product(*param_values):
        params = dict(zip(param_names, param_))

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=params["tfidfvectorizer__ngram_range"],
            max_df=params["tfidfvectorizer__max_df"],
            min_df=params["tfidfvectorizer__min_df"],
        )

        nb = MultinomialNB(alpha=params["nb__alpha"])
        pipeline = make_pipeline(vectorizer, nb)
        pipeline.fit(train_x, train_y)
        preds = pipeline.predict(val_x)

        acc = accuracy_score(val_y, preds)
        f1 = f1_score(val_y, preds, average="macro")
        recall = recall_score(val_y, preds, average="macro")

        all_results.append((params, acc, f1, recall))
        print(f"Params tried: {params}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}\n")

        if acc > best_score:
            best_score = acc
            best_params = params

    print("Best Parameter Combination:")
    print(best_params)
    print(f"Best Accuracy: {best_score:.4f}")

    print("\n Results Summary:")
    for i, (params, acc, f1, recall) in enumerate(all_results, 1):
        print(f"{i}. {params} → Acc: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

    return best_params, best_score, all_results


def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """
    Plots and saves a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        filename (str): Filename to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close(fig)
