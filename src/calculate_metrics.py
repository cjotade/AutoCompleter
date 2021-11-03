import os
import re
import json
import numpy as np
from fast_autocomplete import AutoComplete
import matplotlib.pyplot as plt

from utils import get_words, get_lengths_found
from ModelManager import ModelManager, get_confusion_matrix, plot_confusion_matrix, get_performance_metrics, calculate_metrics_sep

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
    m = create_model(data_train, spanish_cnt_words=spanish_cnt_words)

    # Predict params
    params = {
        "max_cost": 1000,
        "size_search": 1000,
        "max_dist_threshold": 0.4,
        "scale_mode": "empirical",
        "no_response_use_sm_priors": False,
    }

    # Calculate found_words and non_found_words
    found_words, non_found_words, words_predicted_by_line = m.fill_metrics(
        data_test,
       **params
    )

    # Test for single found_words and non_found_words
    tp, fp, tn, fn, info_conf = get_confusion_matrix(found_words, non_found_words)

    #precision, recall, f1_score, acc, fpr, acc_rate, sug_rate = get_performance_metrics(tp, fp, tn, fn)
    acc_rate, sug_rate = get_performance_metrics(tp, fp, tn, fn)

    print(f"acc_rate: {acc_rate:.3f}")
    print(f"sug_rate: {sug_rate:.3f}")
    print(tp, fp)
    print(tn, fn)

    #plot_confusion_matrix(tp, fp, tn, fn, return_norm=False)
    plot_confusion_matrix(tp, fp, tn, fn)

    # Separation by incomplete_word length
    words_sep = calculate_metrics_sep(found_words, non_found_words)
    len_sel_words = words_sep["len_sel_words"]
    acc_rates = words_sep["acc_rates"]
    sug_rates = words_sep["sug_rates"]
    # Plot by len_word
    for i, (s_rate, a_rate) in enumerate(zip(sug_rates, acc_rates)):
        plt.scatter(s_rate, a_rate, color=f"C{i}", label=f"len_word:{len_sel_words[i]}")
        if i > 8:
            break    
    plt.xlabel("suggestion rates", fontsize=16)
    plt.ylabel("accuracy rates", fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.title(f"max_cost:{params.get('max_cost')} - size_search:{params.get('size_search')} - max_dist_thresh:{params.get('max_dist_threshold')}")
    plt.show()

    len_word = 7
    fw_separated_lens = get_lengths_found(found_words)[len_word]
    nfw_separated_lens = get_lengths_found(non_found_words)[len_word]
    tp_s, fp_s, tn_s, fn_s, _ = get_confusion_matrix(fw_separated_lens, nfw_separated_lens)
    acc_rate_s, sug_rate_s = get_performance_metrics(tp_s, fp_s, tn_s, fn_s)
    print(f"acc_rate: {acc_rate_s:.2f}")
    print(f"sug_rate: {sug_rate_s:.2f}")
    print(tp_s, fp_s)
    print(tn_s, fn_s)

    print(json.dumps(fw_separated_lens, indent=4))