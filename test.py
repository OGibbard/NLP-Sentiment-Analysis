import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nlp_functions import *



word_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

training_size = 0.8
validation_size = 0.1

train_set, val_set, test_set = dataset_partitioning(training_size, validation_size, dataset="coursework")

train_reviews = [r[1] for r in train_set]
train_scores  = [r[0] for r in train_set]

train_raw_features, vocab = feature_generation(train_reviews, train_scores, normalization="lemmatize", stop_words_enabled=True, trigrams_enabled=True)

for feature, count in vocab:  # vocab returned from feature_generation()
    if "_" not in feature:
        word_counts[feature] = count
    else:
        parts = feature.split("_")
        if len(parts) == 2:
            bigram_counts[feature] = count
        elif len(parts) == 3:
            trigram_counts[feature] = count

def plot_frequency(counter, title, save_as, vline_x=None):
    items = counter.most_common()
    frequencies = [freq for _, freq in items]
    ranks = range(1, len(frequencies) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".", linestyle="none")
    plt.title(title)
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True, which="both", ls="--")
    
    if vline_x:
        plt.axvline(x=vline_x, color='r', linestyle='--', label=f'Cutoff ({vline_x})')
        plt.legend()

    plt.savefig(save_as, dpi=300)
    plt.close()
    print(f"Saved: {save_as}")

plot_frequency(word_counts,
               "Word Frequency Distribution",
               "word_frequency.png",
               vline_x=5000)

plot_frequency(bigram_counts,
               "PoS Bigram Frequency Distribution",
               "pos_bigram_frequency.png",
               vline_x=100)

plot_frequency(trigram_counts,
               "PoS Trigram Frequency Distribution",
               "pos_trigram_frequency.png",
               vline_x=200)
