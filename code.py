import numpy as np
import os
import random
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball, porter, WordNetLemmatizer
from collections import Counter
import string
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

start_time = time.time()


np.random.seed(5), random.seed(5)

# Split training/validation/testing 80/10/10
def dataset_partitioning(training, validation, dataset="test"):

    pos_folder_path = "aclImdb/train/pos/"
    neg_folder_path = "aclImdb/train/neg/"

    pos_files = [f for f in os.listdir(pos_folder_path)]
    neg_files = [f for f in os.listdir(neg_folder_path)]
    
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    
    limit_per_class = None
    if dataset == "test":
        limit_per_class = 8 # 16 total
    elif dataset == "coursework":
        limit_per_class = 2000 # 4000 total
    elif dataset == "full":
        limit_per_class = len(pos_files) # All
        
    if limit_per_class:
        pos_files = pos_files[:limit_per_class]
        neg_files = neg_files[:limit_per_class]

    pos_training = []
    pos_validation = []
    pos_testing = []
    
    n_pos = len(pos_files)
    n_train = int(n_pos * training)
    n_val = int(n_pos * validation)
    
    for path in pos_files[:n_train]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_training.append((rating.split(".")[0], nltk.sent_tokenize(review)))
    
    for path in pos_files[n_train:n_train+n_val]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_validation.append((rating.split(".")[0], nltk.sent_tokenize(review)))

    for path in pos_files[n_train+n_val:]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_testing.append((rating.split(".")[0], nltk.sent_tokenize(review)))


    neg_training = []
    neg_validation = []
    neg_testing = []
    
    n_neg = len(neg_files)
    n_train_neg = int(n_neg * training)
    n_val_neg = int(n_neg * validation)
    
    for path in neg_files[:n_train_neg]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_training.append((rating.split(".")[0], nltk.sent_tokenize(review)))

    for path in neg_files[n_train_neg:n_train_neg+n_val_neg]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_validation.append((rating.split(".")[0], nltk.sent_tokenize(review)))

    for path in neg_files[n_train_neg+n_val_neg:]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_testing.append((rating.split(".")[0], nltk.sent_tokenize(review)))

    training_dataset = pos_training + neg_training
    validation_dataset = pos_validation + neg_validation
    testing_dataset = pos_testing + neg_testing

    return training_dataset, validation_dataset, testing_dataset

# Split words/bigrams/trigrams/pos-bigrams/pos-trigrams 10000/1200/500/200/100
def feature_generation(reviews, scores):

    word_vocab = []
    bigram_vocab = []
    trigram_vocab = []
    pos_bigram_vocab = []
    pos_trigram_vocab = []

    features = []

    for k in range(len(reviews)):

        review = reviews[k]
        score = scores[k]

        words = []
        useful_bigrams = []
        useful_trigrams = []

        pos_bigrams = []
        pos_trigrams = []


        for line in review:

            #Tokenization and tagging
            remove_tokens = ['<','br','>','/','(',')']
            tokens = [token for token in word_tokenize(line) if token not in remove_tokens]
            tagged = nltk.pos_tag(tokens)


            #Stemming
            snowball_stemmer = snowball.SnowballStemmer("english")
            # porter_stemmer = porter.PorterStemmer()
            stemmed_tokens = [(snowball_stemmer.stem(w), t) for (w, t) in tagged]


            #Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [(lemmatizer.lemmatize(w), t) for (w, t) in tagged]
            lemmatization = True
            if lemmatization:
                used_tokens = lemmatized_tokens
            else:
                used_tokens = stemmed_tokens


            #Remove stop words
            stop_words = True
            if stop_words:
                filtered_tokens = [word for word in used_tokens if not word[0].lower() in stopwords.words('english')]
            else:
                filtered_tokens = used_tokens

            #Clean tokens of useless ones
            cleaned_tokens = []

            
            for token, tag in filtered_tokens:
                token = token.lower()
                if all(c in string.punctuation for c in token):
                    continue

                elif token in {"<", ">", "br", "/"}: #HTML
                    continue

                elif token.isdigit(): #Number
                    continue

                elif token in {"ca", "n't", "'d", "'s"}: #Contractions
                    continue

                elif len(token) == 1 and token not in ['a','i']: #Single characters that aren't words
                    continue

                elif tag in {"NNP", "NNPS",""}: #Remove proper nouns and untagged tokens
                    continue

                else:
                    cleaned_tokens.append((token, tag))

            #Bigrams and Trigrams
            bigrams = [cleaned_tokens[i:i+2] for i in range(len(cleaned_tokens)-1)]
            trigrams = [cleaned_tokens[i:i+3] for i in range(len(cleaned_tokens)-2)]

            Ngrams = bigrams + trigrams

            for gram in bigrams:
                useful_bigrams.append(f"{gram[0][0]}_{gram[1][0]}")
                pos_bigrams.append(f"{gram[0][1]}_{gram[1][1]}")

            for gram in trigrams:
                useful_trigrams.append(f"{gram[0][0]}_{gram[1][0]}_{gram[2][0]}")
                pos_trigrams.append(f"{gram[0][1]}_{gram[1][1]}_{gram[2][1]}")

            # Vocab size
            for word in cleaned_tokens:
                words.append(word[0])

            # result.append()

        features.append({'label' : int(score), 'features' : words + useful_bigrams + useful_trigrams + pos_bigrams + pos_trigrams})

        for feature in words:
            word_vocab.append(feature)
        for feature in useful_bigrams:
            bigram_vocab.append(feature)
        for feature in useful_trigrams:
            trigram_vocab.append(feature)
        for feature in pos_bigrams:
            pos_bigram_vocab.append(feature)
        for feature in pos_trigrams:
            pos_trigram_vocab.append(feature)

    word_vocab_size = 10000
    bigram_vocab_size = 1200
    trigram_vocab_size = 500
    pos_bigram_vocab_size = 200
    pos_trigram_vocab_size = 100


    vocab = Counter(word_vocab).most_common(word_vocab_size) + Counter(bigram_vocab).most_common(bigram_vocab_size) + Counter(trigram_vocab).most_common(trigram_vocab_size) + Counter(pos_bigram_vocab).most_common(pos_bigram_vocab_size) + Counter(pos_trigram_vocab).most_common(pos_trigram_vocab_size)
    return features, vocab
            
def feature_normalisation(features, vocab, method='vector-length'):

    feature_vectors = []    

    if method == 'vector-length':
        # Vector Length Normalisation (Unit Vector)
        for k in range(len(features)):
            vocab_vector = {a[0] : 0 for a in vocab}
            current_features = features[k]['features']
            
            # Raw counts
            for word in current_features:
                if word in vocab_vector:
                    vocab_vector[word] += 1
            
            #L2 norm
            norm = np.sqrt(sum(val**2 for val in vocab_vector.values()))
            
            if norm:
                for word in vocab_vector:
                    vocab_vector[word] /= norm
            
            feature_vectors.append((features[k]['label'], vocab_vector))

    elif method == 'tf-idf':

        doc_freq = {a[0]: 0 for a in vocab}
        for k in range(len(features)):
            current_features = set(features[k]['features'])
            for word in current_features:
                 if word in doc_freq:
                    doc_freq[word] += 1
                    
        N = len(features)
        
        for k in range(len(features)):
            vocab_vector = {a[0] : 0 for a in vocab}
            current_features = features[k]['features']
            
            tf_counter = Counter(current_features)
            
            for word, count in tf_counter.items():
                if word in vocab_vector:
                    tf = count / len(current_features)
                    idf = np.log(N / (doc_freq[word] + 1)) 
                    vocab_vector[word] = tf * idf

            feature_vectors.append((features[k]['label'], vocab_vector))
            
    elif method == 'ppmi':

        total_term_counts = {a[0]: 0 for a in vocab}
        total_words = 0
        
        for k in range(len(features)):
            current_features = features[k]['features']
            for word in current_features:
                if word in total_term_counts:
                    total_term_counts[word] += 1
                    total_words += 1
                    
        for k in range(len(features)):
            vocab_vector = {a[0] : 0 for a in vocab}
            current_features = features[k]['features']
            doc_length = len(current_features)
            
            tf_counter = Counter(current_features)
            
            for word, count in tf_counter.items():
                if word in vocab_vector:
                    n_ij = count
                    n_i = total_term_counts[word]
                    n_j = doc_length
                    N_total = total_words
                    
                    if n_i > 0 and n_j > 0:
                        pmi = np.log2((n_ij * N_total) / (n_i * n_j))
                        ppmi = max(pmi, 0)
                        vocab_vector[word] = ppmi
            
            feature_vectors.append((features[k]['label'], vocab_vector))

    return feature_vectors

def feature_selection(method='vector_length', dataset="test"):

    training_size = 0.8
    validation_size = 0.1

    # method = 'ppmi' #vector-length/tf-idf/ppmi

    training_dataset, validation_dataset, testing_dataset = dataset_partitioning(training_size, validation_size, dataset=dataset)
    
    # Extract reviews and scores for each split
    train_reviews = [review[1] for review in training_dataset]
    train_scores = [review[0] for review in training_dataset]
    
    val_reviews = [review[1] for review in validation_dataset]
    val_scores = [review[0] for review in validation_dataset]
    
    test_reviews = [review[1] for review in testing_dataset]
    test_scores = [review[0] for review in testing_dataset]

    print("Training dataset size: ", len(training_dataset))
    print("Validation dataset size: ", len(validation_dataset))
    print("Testing dataset size: ", len(testing_dataset))
    
    # Generate features and vocab ONLY from training data
    train_features_raw, vocab = feature_generation(train_reviews, train_scores)
    print("Vocab: ", len(vocab))

    # We need to generate feature lists for val and test using the SAME vocab
    # Reuse feature_generation logic but force it to use the existing vocab? 
    # Actually feature_generation returns the features list and the vocab. 
    # The current feature_generation builds vocab from the input reviews.
    # We need a way to extract features from val/test without adding to vocab.
    # For simplicity in this specific codebase structure, we can just run feature_generation
    # on val/test but IGNORE the returned vocab, and then use feature_normalisation
    # which takes the 'vocab' list as input to build the vectors.
    
    val_features_raw, _ = feature_generation(val_reviews, val_scores)
    test_features_raw, _ = feature_generation(test_reviews, test_scores)

    # Normalize features using the TRAINING vocab
    train_vectors = feature_normalisation(train_features_raw, vocab, method=method)
    val_vectors = feature_normalisation(val_features_raw, vocab, method=method)
    test_vectors = feature_normalisation(test_features_raw, vocab, method=method)

    return train_vectors, val_vectors, test_vectors

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.conditional_probs = {}
        self.vocab = set()
        self.classes = set()

    def fit(self, X, y):
        # X is a list of dictionaries (word -> count/value)
        # y is a list of labels
        
        n_docs = len(X)
        self.classes = set(y)
        self.vocab = set()
        
        # Count documents per class
        class_counts = Counter(y)
        
        # Calculate Priors: P(c) = N_c / N
        for c in self.classes:
            self.priors[c] = np.log(class_counts[c] / n_docs)
            
        # Count words per class
        # term_counts[class][word]
        term_counts = {c: Counter() for c in self.classes}
        total_terms_in_class = {c: 0 for c in self.classes}
        
        for i in range(n_docs):
            c = y[i]
            features = X[i]
            for word, count in features.items():
                # For Multinomial NB, we use the counts
                # If features are normalized (floats), we might treat them as weights or just counts
                # The prompt asks for "relevant mathematical formulae". 
                # Standard Multinomial NB works with integer counts.
                # If input is normalized (e.g. TF-IDF), it's often still used with NB but technically violates assumptions.
                # We will assume X contains counts or we treat values as such.
                term_counts[c][word] += count
                total_terms_in_class[c] += count
                self.vocab.add(word)
                
        vocab_size = len(self.vocab)
        
        # Calculate Conditional Probabilities: P(w|c) = (count(w, c) + 1) / (count(c) + |V|)
        # Using Laplace Smoothing (add-1)
        for c in self.classes:
            self.conditional_probs[c] = {}
            denominator = total_terms_in_class[c] + vocab_size
            
            for word in self.vocab:
                numerator = term_counts[c].get(word, 0) + 1
                self.conditional_probs[c][word] = np.log(numerator / denominator)
                
    def predict(self, X):
        predictions = []
        for features in X:
            class_scores = {c: self.priors[c] for c in self.classes}
            
            for word, count in features.items():
                if word in self.vocab:
                    for c in self.classes:
                        class_scores[c] += self.conditional_probs[c].get(word, 0) * count
            
            # Choose class with max score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
        return predictions

def classification(train_vectors, val_vectors, test_vectors):
        
    # Prepare X and y
    X_train = [d[1] for d in train_vectors]
    y_train = [d[0] for d in train_vectors]
    
    X_val = [d[1] for d in val_vectors]
    y_val = [d[0] for d in val_vectors]
    
    X_test = [d[1] for d in test_vectors]
    y_test = [d[0] for d in test_vectors]
    
    print(f"X_train size: {len(X_train)}")
    print(f"X_test size: {len(X_test)}")
    
    if len(X_train) == 0:
        print("Error: X_train is empty!")
        return

    results = []

    # --- 1. Naive Bayes (From Scratch) ---
    print("\nTraining Naive Bayes (From Scratch)...")
    nb_scratch = NaiveBayesClassifier()
    nb_scratch.fit(X_train, y_train)
    y_pred_scratch = nb_scratch.predict(X_test)
    
    results.append({
        "Model": "Naive Bayes (Scratch)",
        "Accuracy": accuracy_score(y_test, y_pred_scratch),
        "Precision": precision_score(y_test, y_pred_scratch, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_scratch, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred_scratch, average='weighted', zero_division=0)
    })

    # --- Prepare for Sklearn (Vectorization) ---
    # DictVectorizer converts list of dicts to sparse matrix
    v = DictVectorizer(sparse=True)
    X_train_sk = v.fit_transform(X_train)
    X_test_sk = v.transform(X_test)

    # --- 2. Naive Bayes (Sklearn) ---
    print("Training Naive Bayes (Sklearn)...")
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train_sk, y_train)
    y_pred_nb = clf_nb.predict(X_test_sk)
    
    results.append({
        "Model": "Naive Bayes (Sklearn)",
        "Accuracy": accuracy_score(y_test, y_pred_nb),
        "Precision": precision_score(y_test, y_pred_nb, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_nb, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)
    })

    # --- 3. Decision Tree ---
    print("Training Decision Tree...")
    clf_dt = DecisionTreeClassifier(random_state=5)
    clf_dt.fit(X_train_sk, y_train)
    y_pred_dt = clf_dt.predict(X_test_sk)
    
    results.append({
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(y_test, y_pred_dt),
        "Precision": precision_score(y_test, y_pred_dt, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_dt, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    })

    # --- 4. Linear Model (Logistic Regression) ---
    print("Training Linear Model (Logistic Regression)...")
    clf_lr = LogisticRegression(max_iter=1000, random_state=5)
    clf_lr.fit(X_train_sk, y_train)
    y_pred_lr = clf_lr.predict(X_test_sk)
    
    results.append({
        "Model": "Linear Model",
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    })

    # --- Print Results ---
    print("\n" + "="*85)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 85)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']:<10.4f} | {res['Precision']:<10.4f} | {res['Recall']:<10.4f} | {res['F1']:<10.4f}")
    print("="*85 + "\n")

np.random.seed(5), random.seed(5)

method = 'ppmi' #vector-length/tf-idf/ppmi

dataset = "coursework"

train_vectors, val_vectors, test_vectors = feature_selection(method, dataset)

classification(train_vectors, val_vectors, test_vectors)

print(f"Running took {time.time() - start_time} seconds.")