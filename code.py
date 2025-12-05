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

start_time = time.time()


np.random.seed(5), random.seed(5)

full_dataset = False

# Split training/validation/testing 80/10/10 10000/1250/1250
def dataset_partitioning(training, validation):

    if full_dataset:
        pos_folder_path = "aclimdb/train/pos/"
        neg_folder_path = "aclimdb/train/neg/"
    else:
        pos_folder_path = "testDB/pos/"
        neg_folder_path = "testDB/neg/"

    pos_files = [f for f in os.listdir(pos_folder_path)]
    random.shuffle(pos_files)

    pos_training = []
    pos_validation = []
    pos_testing = []
    for path in pos_files[:int(len(pos_files)*training)]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_training.append((rating.split(".")[0], review.split(".")))
    
    for path in pos_files[int(len(pos_files)*training):int(len(pos_files)*(training+validation))]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_validation.append((rating.split(".")[0], review.split(".")))

    for path in pos_files[int(len(pos_files)*(training+validation)):]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_testing.append((rating.split(".")[0], review.split(".")))


    neg_files = [f for f in os.listdir(neg_folder_path)]
    random.shuffle(neg_files)

    neg_training = []
    neg_validation = []
    neg_testing = []
    for path in neg_files[:int(len(neg_files)*training)]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_training.append((rating.split(".")[0], review.split(".")))

    for path in neg_files[int(len(neg_files)*training):int(len(neg_files)*(training+validation))]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_validation.append((rating.split(".")[0], review.split(".")))

    for path in neg_files[int(len(neg_files)*(training+validation)):]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_testing.append((rating.split(".")[0], review.split(".")))

    training_dataset = pos_training + neg_training
    validation_dataset = pos_validation + neg_validation
    testing_dataset = pos_testing + neg_testing

    return training_dataset, validation_dataset, testing_dataset

# Split words/bigrams/trigrams 88/10/2 17600/2000/400
def feature_generation(reviews, scores):

    word_vocab_size = 17600
    word_vocab = []

    bigram_vocab_size = 2000
    bigram_vocab = []

    trigram_vocab_size = 400
    trigram_vocab = []


    features = []

    for k in range(len(reviews)):

        review = reviews[k]
        score = scores[k]

        words = []
        useful_bigrams = []
        useful_trigrams = []

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
            bigrams = [cleaned_tokens[k:k+2] for k in range(len(cleaned_tokens)-1)]
            trigrams = [cleaned_tokens[k:k+3] for k in range(len(cleaned_tokens)-2)]

            Ngrams = bigrams + trigrams


            # Noun tag and verb tag phrases
            nouns = ['NN', 'NNS']
            verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            adjectives = ['JJ', 'JJR', 'JJS']
            adverbs = ['RB', 'RBR', 'RBS']

            bigram_phrase_types = [[a, n] for a in adjectives for n in nouns] + [[v, a] for v in verbs for a in adverbs] + [[a, v] for v in verbs for a in adverbs]
            trigram_phrase_types = [[a,a2,n] for a in adjectives for a2 in adjectives for n in nouns] + [[a,a2,n] for a in adverbs for a2 in adjectives for n in nouns] + [[a,n,n2] for a in adjectives for n in nouns for n2 in nouns] + [[n,v,a] for n in nouns for v in verbs for a in adjectives] + [[v,a,a2] for v in verbs for a in adverbs for a2 in adjectives] + [[a,a2,a3] for a in adverbs for a2 in adverbs for a3 in adjectives]


            for gram in bigrams:
                if [word[1] for word in gram] in bigram_phrase_types:
                    gram = gram[0][0] + "_" + gram[1][0]
                    useful_bigrams.append(gram)

            for gram in trigrams:
                if [word[1] for word in gram] in trigram_phrase_types:
                    gram = gram[0][0] + "_" + gram[1][0] + "_" + gram[2][0]
                    useful_trigrams.append(gram)

            # Vocab size
            for word in cleaned_tokens:
                words.append(word[0])

            # result.append()

        features.append({'label' : int(score), 'features' : words + useful_bigrams + useful_trigrams})

        for feature in words:
            word_vocab.append(feature)
        for feature in useful_bigrams:
            bigram_vocab.append(feature)
        for feature in useful_trigrams:
            trigram_vocab.append(feature)


    vocab = Counter(word_vocab).most_common(word_vocab_size) + Counter(bigram_vocab).most_common(bigram_vocab_size) + Counter(trigram_vocab).most_common(trigram_vocab_size)

    return features, vocab
            
def feature_normalisation(features, vocab, method='vector_length'):

    feature_vectors = []    

    if method == 'vector_length':
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

    elif method == 'tfidf':

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

def feature_selection():

    global full_dataset
    full_dataset = False

    training_size = 0.8
    validation_size = 0.1

    method = 'ppmi' #vector-length/tf-idf/ppmi

    training_dataset, validation_dataset, testing_dataset = dataset_partitioning(training_size, validation_size)
    reviews = [review[1] for review in training_dataset]
    scores = [review[0] for review in training_dataset]

    print("Training dataset size: ", len(training_dataset))
    print("Validation dataset size: ", len(validation_dataset))
    print("Testing dataset size: ", len(testing_dataset))
    
    features, vocab = feature_generation(reviews, scores)

    print("Vocab: ", len(vocab))

    feature_vectors = feature_normalisation(features, vocab, method = method)


    return feature_vectors

def classification():
    pass

np.random.seed(5), random.seed(5)

complete_features = feature_selection()

print(complete_features)