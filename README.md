# AfriSentiment: Twi Language Sentiment Analysis

A multilingual NLP project for sentiment analysis of Twi language text using various machine learning approaches including Naive Bayes, TF-IDF, and Logistic Regression.

## Project Overview

AfriSentiment performs sentiment analysis on Twi language texts using the Masakhane AfriSenti dataset. The project implements and compares multiple text classification approaches:

- Naive Bayes (77% accuracy)
- TF-IDF with Multinomial NB (80% accuracy)
- Logistic Regression (79% accuracy)

## Project Structure

```
afrisentiment/
├── data/
│   ├── raw/                    # Original Masakhane AfriSenti dataset
│   └── preprocessed/           # Preprocessed and cleaned data
├── notebooks/
│   ├── exploration.ipynb       # Data analysis & visualization
│   ├── naive_bayes.ipynb      # Naive Bayes implementation
│   ├── tf_idf.ipynb           # TF-IDF vectorization & classification
│   └── logistic_regression.ipynb    # Logistic regression model
├── results/
│   └── plots/                  # Model performance visualizations
├── src/
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py        # Text preprocessing utilities
│   └── utils.py               # Helper functions
├── requirements.txt           # Project dependencies
├── setup.py                  # Package setup configuration
└── README.md                 # Project documentation
```

## Dataset

The project uses the [Masakhane AfriSenti Twi dataset](https://huggingface.co/datasets/masakhane/afrisenti/viewer/twi) which contains:

- Binary sentiment labels (positive/negative)
- Training set: 2,959 samples
  - Positive: 55.56%
  - Negative: 44.44%
- Validation and test sets for model evaluation

You can access and download the dataset directly from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("masakhane/afrisenti", "twi")
```

## Features

- Text preprocessing pipeline optimized for Twi language
- Multiple model implementations:
  - Naive Bayes classifier
  - TF-IDF with Multinomial NB
  - Logistic Regression
- Cross-validation for hyperparameter tuning
- Evaluation metrics including accuracy, F1-score, and confusion matrices
- Data visualization and exploratory analysis

## Model Performance

| Model                  | Accuracy | Macro F1-Score |
| ---------------------- | -------- | -------------- |
| TF-IDF + MultinomialNB | 80.07%   | 79.63%         |
| Logistic Regression    | 79.00%   | 78.00%         |
| Naive Bayes            | 77.00%   | 77.00%         |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kcnewman/afrisentiment.git
cd afrisentiment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:

```python
from preprocessing import DatasetPreprocessor

preprocessor = DatasetPreprocessor(
    input_dir="data/raw",
    output_dir="data/preprocessed"
)
preprocessor.preprocess_all()
```

2. Model Training & Evaluation:

```python
# Example using TF-IDF model
from utils import build_tfidf
model = build_tfidf(alpha=0.4, ngram_range=(1, 2))
model.fit(train_x, train_y)
predictions = model.predict(test_x)
```

## Requirements

- pandas
- notebook
- numpy
- scikit-learn
- matplotlib
- datasets

## Author

- Kelvin Newman (newmankelvin14@gmail.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Masakhane AfriSenti](https://github.com/masakhane-io/masakhane-sentiment) dataset creators and contributors
- [African NLP community](https://www.masakhane.io/) - Masakhane Initiative
- [Twi Language Resources](https://github.com/twi-digital-language-resources) - For language specific tools and resources
