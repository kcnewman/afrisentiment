# Naive Bayes Pipeline

import numpy as np

from src.utils import (
    build_freqs,
    train_naive_bayes,
    cross_validation,
    load_data,
    retrain_final_model,
    evaluate_model,
    print_sample_prediction,
)


def main():
    train_df, val_df, test_df, encoder = load_data()

    train_x, train_y = train_df["tweet"], train_df["sentiment"]
    val_x, val_y = val_df["tweet"], val_df["sentiment"]
    test_x, test_y = test_df["tweet"], test_df["sentiment"]

    freqs = build_freqs(train_x, train_y)
    logprior, loglikelihood, vocab, classes = train_naive_bayes(freqs, train_x, train_y)

    print_sample_prediction("3kom", logprior, loglikelihood, vocab, classes)

    alphas = [0.1, 0.01, 0.2, 0.02, 0.3, 0.03, 0.4, 0.04, 0.5, 0.05]
    best_alpha, scores = cross_validation(train_x, train_y, val_x, val_y, alphas)
    print(f"Best alpha: {best_alpha} | Scores: {scores}")

    full_train_x = np.concatenate([train_x, val_x])
    full_train_y = np.concatenate([train_y, val_y])
    logprior_f, loglikelihood_f, vocab_f, classes_f = retrain_final_model(
        full_train_x, full_train_y, best_alpha
    )

    evaluate_model(
        test_x, test_y, logprior_f, loglikelihood_f, vocab_f, classes_f, encoder
    )


if __name__ == "__main__":
    main()
