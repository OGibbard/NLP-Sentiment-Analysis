from nlp_functions import *

start_time = time.time()

np.random.seed(6), random.seed(6)

dataset = "coursework"

training_size = 0.8
validation_size = 0.1

train_set, val_set, test_set = dataset_partitioning(training_size, validation_size, dataset=dataset)


# method, normalization, stop_words_enabled, trigrams_enabled = pipeline_tuning((train_set, val_set, test_set))
# Best Pipeline: ('tf-idf', 'none', True, False) (Val Acc: 0.8675)

method = 'tf-idf'
normalization = 'none'
stop_words_enabled = True
trigrams_enabled = False

vocab_values = feature_selection(train_set, val_set, test_set, normalization, stop_words_enabled, trigrams_enabled, method)

classification(vocab_values['values'])

print(f"Running took {time.time() - start_time} seconds.")