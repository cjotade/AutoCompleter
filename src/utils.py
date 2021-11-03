import os 
import json
from nltk import everygrams

def get_words(data, n_grams=3):
    """
    words: dict with all words and ngrams
    ngrams_cnt_words: dict with all words and ngrams with frequency
    cnt_words: dict with all words with frequency
    """
    words, cnt_words, ngrams_words, ngrams_cnt_words = {}, {}, {}, {}
    words_per_line = []
    chars_per_words_in_line = []
    for i, line in enumerate(data):
        line_splitted = line.split()
        words_per_line.append(len(line_splitted))
        # Calculate chars_per_words and cnt_words
        chars_per_words = []
        for w in line_splitted:
            words[w] = {}
            chars_per_words.append(len(w))
            if cnt_words.get(w, False):
                cnt_words[w] += 1
            else:
                cnt_words[w] = 1
        chars_per_words_in_line.append(chars_per_words)
        # Calculate everygrams and store every phrase in words, also add ngrams_cnt_words
        threegrams = everygrams(line_splitted, max_len=n_grams)
        for threegram in threegrams:
            phrase = " ".join(threegram)
            ngrams_words[phrase] = {}
            if ngrams_cnt_words.get(phrase, False):
                ngrams_cnt_words[phrase] += 1
            else:
                ngrams_cnt_words[phrase] = 1
    return words, cnt_words, ngrams_words, ngrams_cnt_words, words_per_line, chars_per_words_in_line

def get_lengths_found(words):
    """
    words: List of dicts with incomplete_words and words at least
    Return a dictionary with keys as the word len and values a list with the response dictionary
    """
    words_select = {}
    for ws_dict in words:
        word_len = len(ws_dict["incomplete_word"])
        words_select.setdefault(word_len, []).append(ws_dict)
    return words_select

def load_words(save_folder, key):
    f1 = open(os.path.join(save_folder, f'foundwords_{key}.json'), 'r')
    found_words = json.load(f1)
    f2 = open(os.path.join(save_folder, f'nonfoundwords_{key}.json'), 'r')
    non_found_words = json.load(f2)
    f3 = open(os.path.join(save_folder, f'wordspredictedbyline_{key}.json'), 'r')
    words_predicted_by_line = json.load(f3)
    return found_words, non_found_words, words_predicted_by_line

def get_keys_startswith_dict(dictionary, key_start):
    new_dict = {}
    for k, v in dictionary.items():
        if k.startswith(key_start):
            new_dict[k] = v
    return new_dict
