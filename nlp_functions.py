import numpy as np
import os
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import snowball, porter, WordNetLemmatizer
from collections import Counter
import string
import time
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import SGDClassifier

#Split training/validation/testing 80/10/10
def dataset_partitioning(training, validation, dataset="test"):

    pos_folder_path = "aclImdb/train/pos/"
    neg_folder_path = "aclImdb/train/neg/"

    pos_files = [f for f in os.listdir(pos_folder_path)]
    neg_files = [f for f in os.listdir(neg_folder_path)]
    
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    
    limit_per_class = None
    if dataset == "test":
        limit_per_class = 10 # 20 total
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

        pos_training.append((1, review))
    
    for path in pos_files[n_train:n_train+n_val]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_validation.append((1, review))

    for path in pos_files[n_train+n_val:]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_testing.append((1, review))


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
        neg_training.append((0, review))

    for path in neg_files[n_train_neg:n_train_neg+n_val_neg]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_validation.append((0, review))

    for path in neg_files[n_train_neg+n_val_neg:]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_testing.append((0, review))

    training_dataset = pos_training + neg_training
    validation_dataset = pos_validation + neg_validation
    testing_dataset = pos_testing + neg_testing

    return training_dataset, validation_dataset, testing_dataset


#Split words/bigrams/trigrams/pos-bigrams/pos-trigrams 10000/1200/500/200/100
def feature_generation(reviews, scores, normalization="lemmatize", stop_words_enabled=True, trigrams_enabled=True):

    word_vocab = []
    bigram_vocab = []
    pos_bigram_vocab = []
    trigram_vocab = []
    pos_trigram_vocab = []

    lemmatizer = WordNetLemmatizer()
    snowball_stemmer = snowball.SnowballStemmer("english")
    english_stopwords = set(stopwords.words('english'))

    features = []

    for k in range(len(reviews)):

        review = reviews[k]
        score = scores[k]

        words = []
        useful_bigrams = []
        pos_bigrams = []

        useful_trigrams = []
        pos_trigrams = []

        #Tokenization and tagging
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", review.lower()) #Regex to keep only words
        tagged = nltk.pos_tag(tokens)


        #Stemming
        # porter_stemmer = porter.PorterStemmer()
        stemmed_tokens = [(snowball_stemmer.stem(w), t) for (w, t) in tagged]


        #Lemmatization
        lemmatized_tokens = [(lemmatizer.lemmatize(w), t) for (w, t) in tagged]

        if normalization == "lemmatize":
            used_tokens = lemmatized_tokens
        elif normalization == "stem":
            used_tokens = stemmed_tokens
        elif normalization == "none":
            used_tokens = tagged


        #Remove stop words
        if stop_words_enabled:
            filtered_tokens = [word for word in used_tokens if not word[0].lower() in english_stopwords]
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

            elif len(token) == 1 and token not in ['a','i']: #Single characters except a and i
                continue

            elif tag in {"NNP", "NNPS",""}: #Remove proper nouns and untagged tokens
                continue

            else:
                cleaned_tokens.append((token, tag))

        #Bigrams and Trigrams
        bigrams = [cleaned_tokens[i:i+2] for i in range(len(cleaned_tokens)-1)] #List of bigrams [[(token, tag), (token, tag)], ...]
        if trigrams_enabled:
            trigrams = [cleaned_tokens[i:i+3] for i in range(len(cleaned_tokens)-2)] #List of trigrams [[(token, tag), (token, tag), (token, tag)], ...]

        for gram in bigrams:
            useful_bigrams.append(f"{gram[0][0]}_{gram[1][0]}") #List of bigrams ["word1_word2", ...]
            pos_bigrams.append(f"{gram[0][1]}_{gram[1][1]}") #List of bigrams ["tag1_tag2", ...]

        if trigrams_enabled:
            for gram in trigrams:
                useful_trigrams.append(f"{gram[0][0]}_{gram[1][0]}_{gram[2][0]}") #List of trigrams ["word1_word2_word3", ...]
                pos_trigrams.append(f"{gram[0][1]}_{gram[1][1]}_{gram[2][1]}") #List of trigrams ["tag1_tag2_tag3", ...]

        # Vocab size
        for word in cleaned_tokens:
            words.append(word[0])

        if trigrams_enabled:
            features.append({'label' : int(score), 'features' : words + useful_bigrams + useful_trigrams + pos_bigrams + pos_trigrams})
        else:
            features.append({'label' : int(score), 'features' : words + useful_bigrams + pos_bigrams})

        for feature in words:
            word_vocab.append(feature)
        for feature in useful_bigrams:
            bigram_vocab.append(feature)
        for feature in pos_bigrams:
            pos_bigram_vocab.append(feature)
        if trigrams_enabled:
            for feature in useful_trigrams:
                trigram_vocab.append(feature)
            for feature in pos_trigrams:
                pos_trigram_vocab.append(feature)

    word_vocab_size = 5000
    bigram_vocab_size = 1000
    pos_bigram_vocab_size = 100
    trigram_vocab_size = 100
    pos_trigram_vocab_size = 200


    vocab = Counter(word_vocab).most_common(word_vocab_size) + Counter(bigram_vocab).most_common(bigram_vocab_size) + Counter(pos_bigram_vocab).most_common(pos_bigram_vocab_size)
    if trigrams_enabled:
        vocab += Counter(trigram_vocab).most_common(trigram_vocab_size) + Counter(pos_trigram_vocab).most_common(pos_trigram_vocab_size)
    return features, vocab
            
def feature_normalisation(features, vocab, method='counts'):

    feature_vectors = []

    #counts
    if method == 'counts':
        for k in range(len(features)):
            vocab_vector = {a[0]: 0 for a in vocab}
            current_features = features[k]['features']
            for word in current_features:
                if word in vocab_vector:
                    vocab_vector[word] += 1
            feature_vectors.append((features[k]['label'], vocab_vector))

    #vector-length-normalisation
    elif method == 'vector-length':
        for k in range(len(features)):
            vocab_vector = {a[0]: 0 for a in vocab}
            current_features = features[k]['features']
            for word in current_features:
                if word in vocab_vector:
                    vocab_vector[word] += 1

            norm = np.sqrt(sum(val**2 for val in vocab_vector.values()))
            if norm:
                for word in vocab_vector:
                    vocab_vector[word] /= norm

            feature_vectors.append((features[k]['label'], vocab_vector))

    #tf-idf
    elif method == 'tf-idf':

        #Document frequency
        doc_freq = {a[0]: 0 for a in vocab}
        N = len(features)

        for k in range(N):
            seen = set(features[k]['features'])
            for w in seen:
                if w in doc_freq:
                    doc_freq[w] += 1

        #TF-IDF vector for each document
        for k in range(N):
            vocab_vector = {a[0]: 0 for a in vocab}
            current_features = features[k]['features']
            tf_counter = Counter(current_features)

            for word, count in tf_counter.items():
                if word in vocab_vector:
                    tf = count / len(current_features)
                    idf = np.log((N + 1) / (doc_freq[word] + 1)) + 1 
                    vocab_vector[word] = tf * idf

            feature_vectors.append((features[k]['label'], vocab_vector))

    #PPMI
    elif method == 'ppmi':

        total_counts = {a[0]: 0 for a in vocab}
        total_words = 0

        #Count word occurrences across dataset
        for k in range(len(features)):
            for word in features[k]['features']:
                if word in total_counts:
                    total_counts[word] += 1
                    total_words += 1

        #PPMI vector for each review
        for k in range(len(features)):
            vocab_vector = {a[0]: 0 for a in vocab}
            tf_counter = Counter(features[k]['features'])
            doc_len = len(features[k]['features'])

            for word, count in tf_counter.items():
                if word in vocab_vector:
                    n_ij = count                   #Count of w within d
                    n_i = total_counts[word]       #Count of w across dataset 
                    n_j = doc_len                  #Count of tokens within d
                    N_tot = total_words            #Total tokens in dataset
                    if n_i > 0 and n_j > 0:
                        pmi = np.log2((n_ij * N_tot) / (n_i * n_j))
                        vocab_vector[word] = max(pmi, 0) #Ensure positive

            feature_vectors.append((features[k]['label'], vocab_vector))

    return feature_vectors

def feature_selection(train_set, val_set, test_set, normalization="lemmatize", stop_words_enabled=True, trigrams_enabled=True, method='counts'):

    train_reviews = [r[1] for r in train_set]
    train_scores  = [r[0] for r in train_set]

    val_reviews = [r[1] for r in val_set]
    val_scores  = [r[0] for r in val_set]

    test_reviews = [r[1] for r in test_set]
    test_scores  = [r[0] for r in test_set]

    train_raw_features, vocab = feature_generation(train_reviews, train_scores, normalization=normalization, stop_words_enabled=stop_words_enabled, trigrams_enabled=trigrams_enabled)
    val_raw_features, _  = feature_generation(val_reviews, val_scores, normalization=normalization, stop_words_enabled=stop_words_enabled, trigrams_enabled=trigrams_enabled)
    test_raw_features, _ = feature_generation(test_reviews, test_scores, normalization=normalization, stop_words_enabled=stop_words_enabled, trigrams_enabled=trigrams_enabled)

    train_values = feature_normalisation(train_raw_features, vocab, method=method)
    val_values   = feature_normalisation(val_raw_features, vocab, method=method)
    test_values  = feature_normalisation(test_raw_features, vocab, method=method)

    vocab_values = {"vocab": vocab, "values": (train_values, val_values, test_values)}

    return vocab_values

class MyNaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.priors = {}
        self.conditional_probs = {}
        self.vocab = set()
        self.classes = set()

    def fit(self, X, y):
        #X = list of dictionaries
        #y = list of labels
        
        n_docs = len(X)
        self.classes = set(y)
        self.vocab = set()
        
        class_counts = Counter(y)
        
        #Calculate Priors: P(c) = ln(N_c / N)
        for c in self.classes:
            self.priors[c] = np.log(class_counts[c] / n_docs) #50/50
            
        #Count words for each class
        term_counts = {c: Counter() for c in self.classes}
        total_terms_in_class = {c: 0 for c in self.classes}
        
        for i in range(n_docs):
            c = y[i]
            features = X[i]
            for word, count in features.items():
                term_counts[c][word] += count
                total_terms_in_class[c] += count
                self.vocab.add(word)
                
        vocab_size = len(self.vocab)
        
        #Calculate conditional probs: P(w | c) = ln((count(w, c) + alpha) / (count(c) + alpha * |V|))
        for c in self.classes:
            self.conditional_probs[c] = {}
            denominator = total_terms_in_class[c] + (vocab_size * self.alpha)
            
            for word in self.vocab:
                numerator = term_counts[c].get(word, 0) + self.alpha
                self.conditional_probs[c][word] = np.log(numerator / denominator)
                
    def predict(self, X):
        predictions = []
        for features in X:
            class_scores = {c: self.priors[c] for c in self.classes}
            
            #Calculate scores
            for word, count in features.items():
                if word in self.vocab:
                    for c in self.classes:
                        class_scores[c] += self.conditional_probs[c].get(word, 0) * count
            
            #Choose best class
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
        return predictions

def pipeline_tuning(values):

    results_table = []

    (train_values, val_values, test_values) = values
    
    methods = ['counts', 'vector-length', 'tf-idf', 'ppmi']
    normalizations = ["lemmatize", "stem", "none"]
    booleans = [True, False]

    best_acc = -1
    best_pipeline = None

    for method in methods:
        for normalization in normalizations:
            for stop_words_enabled in booleans:
                for trigrams_enabled in booleans:
                    print(f"\nMethod: {method}, Normalization: {normalization}, Stop Words: {stop_words_enabled}, Trigrams: {trigrams_enabled}")
                    vocab_values = feature_selection(train_values, val_values, test_values, normalization, stop_words_enabled, trigrams_enabled, method)
                    
                    (train_feats, val_feats, test_feats) = vocab_values['values']
                    
                    X_train_loop = [d[1] for d in train_feats]
                    y_train_loop = [d[0] for d in train_feats]
                    
                    X_val_loop = [d[1] for d in val_feats]
                    y_val_loop = [d[0] for d in val_feats]
                    
                    v = DictVectorizer(sparse=True)
                    X_train_vec = v.fit_transform(X_train_loop)
                    X_val_vec = v.transform(X_val_loop)
                    
                    nb = MultinomialNB(alpha=1.0)
                    nb.fit(X_train_vec, y_train_loop)
                    y_pred_val = nb.predict(X_val_vec)
                    acc = accuracy_score(y_val_loop, y_pred_val)
                    print(f"Val Acc: {acc:.4f}")

                    results_table.append({
                        "Method": method,
                        "Normalization": normalization,
                        "Stop Words": stop_words_enabled,
                        "Trigrams": trigrams_enabled,
                        "Accuracy": acc
                    })

                    if acc > best_acc:
                        best_acc = acc
                        best_pipeline = (method, normalization, stop_words_enabled, trigrams_enabled)

    # Print table after all runs
    print("\n" + "="*100)
    print(f"{'Method':<15} | {'Norm':<12} | {'StopWords':<10} | {'Trigrams':<10} | {'Accuracy':<8}")
    print("-"*100)

    for row in results_table:
        print(f"{row['Method']:<15} | {row['Normalization']:<12} | "
              f"{str(row['Stop Words']):<10} | {str(row['Trigrams']):<10} | "
              f"{row['Accuracy']:<8.4f}")

    print("="*100)
    print(f"\nBest Pipeline: {best_pipeline} (Val Acc = {best_acc:.4f})\n")
    return best_pipeline

def classification(values):

    (train_values, val_values, test_values) = values

    X_train = [d[1] for d in train_values]
    y_train = [d[0] for d in train_values]

    X_val = [d[1] for d in val_values]
    y_val = [d[0] for d in val_values]

    X_test  = [d[1] for d in test_values]
    y_test = [d[0] for d in test_values]

    results = []

    #My Naive Bayes
    best_acc = -1
    best_alpha = 1.0
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    for alpha in alphas:
        nb = MyNaiveBayesClassifier(alpha=alpha)
        nb.fit(X_train, y_train)
        y_pred_val = nb.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        precision = precision_score(y_val, y_pred_val)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            
    print(f"Best Alpha (Scratch): {best_alpha} (Val Acc: {best_acc:.4f}, Val Precision: {precision:.4f})")
    
    nb_scratch = MyNaiveBayesClassifier(alpha=best_alpha)
    nb_scratch.fit(X_train, y_train)
    y_pred = nb_scratch.predict(X_test)

    results.append({"Model": f"Naive Bayes (Scratch) a={best_alpha}", "Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred)})

    #Vectorize features
    v = DictVectorizer(sparse=True)
    X_train_vec = v.fit_transform(X_train)
    X_val_vec = v.transform(X_val)
    X_test_vec  = v.transform(X_test)

    #Sklearn Naive Bayes
    best_acc = -1
    best_alpha_sk = 1.0
    
    for alpha in alphas:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train_vec, y_train)
        y_pred_val = clf.predict(X_val_vec)
        acc = accuracy_score(y_val, y_pred_val)
        precision = precision_score(y_val, y_pred_val)
        if acc > best_acc:
            best_acc = acc
            best_alpha_sk = alpha

    print(f"Best Alpha (Sklearn): {best_alpha_sk} (Val Acc: {best_acc:.4f}, Val Precision: {precision:.4f})")
    
    nb_sklearn = MultinomialNB(alpha=best_alpha_sk)
    nb_sklearn.fit(X_train_vec, y_train)
    y_pred = nb_sklearn.predict(X_test_vec)

    results.append({"Model": f"Naive Bayes (sklearn) a={best_alpha_sk}", "Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred)})

    #Decision Tree
    best_acc = -1
    best_params_dt = {}
    
    min_samples_splits = [2, 5, 10]
    criterions = ['gini', 'entropy']
    max_depths = [10, 20, 50, None]
    max_features = [1000, 5000, None]
    
    for crit in criterions:
        for mss in min_samples_splits:
            for md in max_depths:
                for mf in max_features:
                    clf = DecisionTreeClassifier(criterion=crit, min_samples_split=mss, max_depth=md, max_features=mf, random_state=5)
                    clf.fit(X_train_vec, y_train)
                    y_pred_val = clf.predict(X_val_vec)
                    acc = accuracy_score(y_val, y_pred_val)
                    precision = precision_score(y_val, y_pred_val)
                    if acc > best_acc:
                        best_acc = acc
                        best_params_dt = {'criterion': crit, 'min_samples_split': mss, 'max_depth': md, 'max_features': mf}
                
    print(f"Best Params (DT): {best_params_dt} (Val Acc: {best_acc:.4f}, Val Precision: {precision:.4f})")

    clf_dt = DecisionTreeClassifier(criterion=best_params_dt['criterion'], min_samples_split=best_params_dt['min_samples_split'], max_depth=best_params_dt['max_depth'], max_features=best_params_dt['max_features'], random_state=5)
    clf_dt.fit(X_train_vec, y_train)
    y_pred = clf_dt.predict(X_test_vec)

    results.append({"Model": "Decision Tree", "Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred)})

    #SGD Linear Model
    best_acc = -1
    
    losses = ['hinge', 'log_loss']
    penalties = ['l2', 'l1', 'elasticnet']
    alphas_sgd = [0.0001, 0.001, 0.01]
    
    for loss in losses:
        for pen in penalties:
            for alpha in alphas_sgd:
                clf = SGDClassifier(loss=loss, penalty=pen, alpha=alpha, random_state=5)
                clf.fit(X_train_vec, y_train)
                y_pred_val = clf.predict(X_val_vec)
                acc = accuracy_score(y_val, y_pred_val)
                precision = precision_score(y_val, y_pred_val)
                if acc > best_acc:
                    best_acc = acc
                    best_params_sgd = {'loss': loss, 'penalty': pen, 'alpha': alpha}

    print(f"Best Params (SGD): {best_params_sgd} (Val Acc: {best_acc:.4f}, Val Precision: {precision:.4f})")

    clf_sgd = SGDClassifier(loss=best_params_sgd['loss'], penalty=best_params_sgd['penalty'], alpha=best_params_sgd['alpha'], random_state=5)
    clf_sgd.fit(X_train_vec, y_train)
    y_pred = clf_sgd.predict(X_test_vec)

    results.append({"Model": "SGD Linear Model", "Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred)})

    print("="*85)
    print(f"{'Model':<35} | {'Accuracy':<10} | {'Precision':<10}")
    print("-"*85)
    for r in results:
        print(f"{r['Model']:<35} | {r['Accuracy']:<10.4f} | {r['Precision']:<10.4f}")
    print("="*85)
