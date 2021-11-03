import os
import json
from fast_autocomplete import AutoComplete

from utils import get_words
from ModelManager import ModelManager

def load_data(data_folder="../data/"):
    filename_train = "ES-es_colloquial_cleaned.txt"
    with open(os.path.join(data_folder, filename_train), 'r') as fp:
        data_train = fp.read().splitlines()

    filename_test = "ES-es_colloquial_01_cleaned.txt"
    with open(os.path.join(data_folder, filename_test), 'r') as fp:
        data_test = fp.read().splitlines()

    filename_spanish = 'spanish_words.json'
    with open(os.path.join(data_folder, filename_spanish), 'r') as fp:
        spanish_cnt_words = json.load(fp)
    
    return data_train, data_test, spanish_cnt_words

def create_model(data, spanish_cnt_words=None):
    # Words and count
    words, cnt_words, ngrams_words, _, _, _ = get_words(data)
    # Suffix Tree Model
    st_autocompleter = AutoComplete(words=ngrams_words)
    # Metrics calculator
    return ModelManager(st_autocompleter, cnt_words, spanish_count_words=spanish_cnt_words)  


if __name__ == "__main__":
    # Load data
    data_train, data_test, spanish_cnt_words = load_data(data_folder="../data/")
    # Load model
    model = create_model(data_train, spanish_cnt_words=spanish_cnt_words)

    # Predict params
    params = {
        "max_cost": 1000,
        "size_search": 1000,
        "max_dist_threshold": 0.4,
        "scale_mode": "empirical",
        "no_response_use_sm_priors": False,
    }

    incomplete_word = "garban"

    # Calculate predictions
    predictions = model.predict(
        incomplete_word=incomplete_word,
        **params
    )

    if predictions:
        best_response, responses, dist_responses = predictions
        print("best_response:", best_response)
    else:
        print("No predictions found")