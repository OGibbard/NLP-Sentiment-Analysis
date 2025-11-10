import numpy as np
import os
import random
from nltk import word_tokenize

np.random.seed(5)
random.seed(5)


# Split training/validation/testing 80/10/10 10000/1250/1250
def dataset_partitioning(training, validation):



    pos_folder_path = "train/pos/"
    pos_files = [f for f in os.listdir(pos_folder_path)]
    random.shuffle(pos_files)

    pos_training = {}
    pos_validation = {}
    pos_testing = {}
    for path in pos_files[:int(len(pos_files)*training)]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_training[num] = (rating.split(".")[0], review)
    
    for path in pos_files[int(len(pos_files)*training):int(len(pos_files)*(training+validation))]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_validation[num] = (rating.split(".")[0], review)

    for path in pos_files[int(len(pos_files)*(training+validation)):]:
        num, rating = path.split("_")
        file = open(pos_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        pos_testing[num] = (rating.split(".")[0], review)



    neg_folder_path = "train/neg/"
    neg_files = [f for f in os.listdir(neg_folder_path)]
    random.shuffle(neg_files)

    neg_training = {}
    neg_validation = {}
    neg_testing = {}
    for path in neg_files[:int(len(neg_files)*training)]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_training[num] = (rating.split(".")[0], review)

    for path in neg_files[int(len(neg_files)*training):int(len(neg_files)*(training+validation))]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_validation[num] = (rating.split(".")[0], review)

    for path in neg_files[int(len(neg_files)*(training+validation)):]:
        num, rating = path.split("_")
        file = open(neg_folder_path + path, "r")
        review = ""
        for text in file:
            review += text
        neg_testing[num] = (rating.split(".")[0], review)



    print(len(pos_training), len(neg_training))
    print(len(pos_validation), len(neg_validation))
    print(len(pos_testing), len(neg_testing))

    return pos_training, pos_validation, pos_testing, neg_training, neg_validation, neg_testing

# def tokenization():


def feature_generation():
    pass

def feature_normalisation():
    pass

def feature_selection():
    pass

def classification():
    pass

pos_training, pos_validation, pos_testing, neg_training, neg_validation, neg_testing = dataset_partitioning(0.8, 0.1)

k=0
while True:
    try:
        print(pos_testing[str(k)])
        break
    except:
        k+=1
        pass