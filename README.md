# NLP Coursework: IMDB Sentiment Classification

This repository contains a full Natural Language Processing pipeline for binary sentiment classification on movie reviews from the Large Movie Review Dataset. The project preprocesses raw IMDB review text, generates lexical and part-of-speech features, compares feature-weighting strategies, tunes classifier hyperparameters, and reports final accuracy and precision on a held-out test split.

The final selected pipeline uses TF-IDF features, stop-word removal, no stemming or lemmatization, and no trigrams. This configuration is then evaluated across a custom Naive Bayes implementation and several scikit-learn models.

## Project Overview

The coursework explores a traditional machine-learning NLP workflow:

1. Load positive and negative IMDB reviews from `aclImdb/train`.
2. Partition the selected dataset into training, validation, and test splits.
3. Tokenise review text and remove noisy tokens such as HTML fragments, punctuation-only tokens, numbers, selected contractions, and proper nouns.
4. Optionally apply stemming, lemmatisation, stop-word removal, and trigram features.
5. Build a vocabulary from frequent words, word n-grams, and POS n-grams.
6. Convert reviews into count, vector-length-normalised, TF-IDF, or PPMI feature vectors.
7. Tune models on the validation split.
8. Report final accuracy and precision on the test split.

## Repository Structure

```text
.
|-- code.py                  # Main experiment script
|-- nlp_functions.py         # Dataset loading, feature engineering, tuning, and classifiers
|-- test.py                  # Frequency-plot generation script
|-- test.ipynb               # Exploratory notebook
|-- environment.yml          # Conda environment definition
|-- aclImdb/                 # IMDB Large Movie Review Dataset
|-- testDB/                  # Small local test dataset
|-- *.png                    # Generated feature-frequency plots
`-- Submission/
    |-- NLP Coursework.pdf   # Final report
    `-- Code/                # Submission copy of the code and generated figures
```

## Dataset

The project uses the Large Movie Review Dataset v1.0 from Maas et al. It contains 50,000 labelled reviews split evenly into positive and negative sentiment classes:

- `aclImdb/train/pos`: 12,500 positive training reviews
- `aclImdb/train/neg`: 12,500 negative training reviews
- `aclImdb/test/pos`: 12,500 positive test reviews
- `aclImdb/test/neg`: 12,500 negative test reviews

This implementation builds its own train/validation/test split from `aclImdb/train`. The dataset size is controlled in `dataset_partitioning()`:

| Mode | Reviews per class | Total reviews | Intended use |
| --- | ---: | ---: | --- |
| `test` | 10 | 20 | Very quick debugging |
| `coursework` | 2,000 | 4,000 | Main coursework experiments |
| `full` | 12,500 | 25,000 | Full training subset |

The default in `code.py` is:

```python
dataset = "coursework"
training_size = 0.8
validation_size = 0.1
```

This creates an 80/10/10 split for training, validation, and testing.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate nlp_coursework
```

The environment installs Python 3.11.9 with NumPy, NLTK, scikit-learn, Matplotlib, and Jupyter Notebook.

NLTK also needs local language resources. If they are not already installed, run:

```bash
python -m nltk.downloader stopwords averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4
```

Depending on the installed NLTK version, either `averaged_perceptron_tagger` or `averaged_perceptron_tagger_eng` may be required for POS tagging.

## Running the Main Experiment

Run the selected coursework pipeline with:

```bash
python code.py
```

The script will:

- load and split the dataset;
- generate word, bigram, and POS-bigram features;
- apply TF-IDF weighting;
- tune model hyperparameters on the validation split;
- evaluate each model on the test split;
- print final accuracy and precision.

The currently selected feature pipeline in `code.py` is:

```python
method = "tf-idf"
normalization = "none"
stop_words_enabled = True
trigrams_enabled = False
```

The script fixes NumPy and Python random seeds for more reproducible dataset shuffling:

```python
np.random.seed(6)
random.seed(6)
```

## Re-running Pipeline Tuning

The repository includes a full pipeline search over:

- feature weighting: counts, vector-length normalisation, TF-IDF, PPMI;
- token normalisation: lemmatisation, stemming, none;
- stop-word removal on/off;
- trigram features on/off.

To run the full search, uncomment this line in `code.py`:

```python
# method, normalization, stop_words_enabled, trigrams_enabled = pipeline_tuning((train_set, val_set, test_set))
```

and comment out or replace the fixed configuration below it.

The best recorded validation result in the current code is:

```text
Best Pipeline: ('tf-idf', 'none', True, False) (Val Acc: 0.8675)
```

## Models Evaluated

`classification()` evaluates four model families:

| Model | Implementation | Tuned parameters |
| --- | --- | --- |
| Naive Bayes | Custom implementation | Laplace smoothing alpha |
| Naive Bayes | `sklearn.naive_bayes.MultinomialNB` | Alpha |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | Criterion, max depth, max features, min samples split |
| SGD Linear Model | `sklearn.linear_model.SGDClassifier` | Loss, penalty, alpha |

The final output is a table with test-set accuracy and precision for each model.

## Feature Engineering Details

Feature generation is implemented in `nlp_functions.py`.

The main preprocessing steps are:

- lowercase regex tokenisation;
- NLTK POS tagging;
- optional Snowball stemming or WordNet lemmatisation;
- optional English stop-word removal;
- filtering of HTML fragments, punctuation-only tokens, numbers, selected contraction fragments, single-character noise, and proper nouns;
- generation of word unigrams, word bigrams, optional word trigrams, POS bigrams, and optional POS trigrams.

The default vocabulary limits are:

| Feature type | Maximum features |
| --- | ---: |
| Word unigrams | 5,000 |
| Word bigrams | 1,000 |
| POS bigrams | 100 |
| Word trigrams | 100 |
| POS trigrams | 200 |

When trigrams are disabled, only word unigrams, word bigrams, and POS bigrams are used.

## Generating Frequency Plots

Run:

```bash
python test.py
```

In its current form, this generates log-log frequency plots and saves:

- `word_frequency.png`
- `pos_bigram_frequency.png`
- `pos_trigram_frequency.png`

The repository also contains existing `bigram_frequency.png` and `trigram_frequency.png` outputs from related feature-frequency analysis.

These plots are useful for inspecting vocabulary cut-off choices and feature-frequency distributions.

## Notes and Limitations

- The code expects the `aclImdb` directory to exist at the repository root.
- `code.py` currently uses the `coursework` subset rather than the full IMDB training set to keep experimentation manageable.
- The full pipeline search can take significantly longer than the default fixed pipeline because it regenerates features for every configuration.
- The custom Naive Bayes classifier is included for coursework comparison against scikit-learn rather than for production use.
- `test.ipynb` is exploratory and may include older experimental cells that do not exactly match the final pipeline.

## Citation

The dataset should be cited as:

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T.
               and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association
               for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150}
}
```
