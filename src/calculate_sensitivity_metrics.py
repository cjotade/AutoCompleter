import os
import json
import numpy as np
from fast_autocomplete import AutoComplete

from utils import get_words, get_lengths_found
from ModelManager import ModelManager, get_confusion_matrix, get_performance_metrics, calculate_metrics_sep

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
    mCalc = create_model(data_train, spanish_cnt_words=spanish_cnt_words)

    # Calculation params
    calculate_sensitivity = True
    calculate_metrics_roc = True
    no_response_use_sm_priors = False
    max_costs = [1000]
    size_searchs = [1000]
    max_dist_thresholds = [0.1, 0.2, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.6, 0.7, 0.8, 0.9]
    scale_mode = "empirical"

    key_format = "maxcost-{}_sizesearch-{}_maxdistthreshold-{}_nrusmp-{}"

    # Store folders
    save_folder = "../sensitivity_suffix_big_spanish"
    calculation_save_folder = os.path.join(save_folder, "metrics_calculation_empirical")
    segmentation_save_folder = os.path.join(save_folder,"segmentation_words_empirical")
    all_metrics_filename = os.path.join(save_folder, f"all_metrics_{scale_mode}_nrusmp-{no_response_use_sm_priors}.npz")

    if not os.path.exists(calculation_save_folder):
        os.makedirs(calculation_save_folder)
        
    if not os.path.exists(segmentation_save_folder):
        os.makedirs(segmentation_save_folder)

    metrics_already_calculated = []

    if calculate_sensitivity:
        for max_cost in max_costs:
            for size_search in size_searchs:
                for max_dist_threshold in max_dist_thresholds:
                    if [max_cost, size_search, max_dist_threshold] in metrics_already_calculated:
                        continue
                    print([max_cost, size_search, max_dist_threshold])
                    found_words, non_found_words, words_predicted_by_line = mCalc.fill_metrics(
                        data_test, 
                        max_cost=max_cost, 
                        size_search=size_search,
                        max_dist_threshold=max_dist_threshold,
                        scale_mode=scale_mode,
                        no_response_use_sm_priors=no_response_use_sm_priors
                    )
                    tp, fp, tn, fn, info_conf = get_confusion_matrix(found_words, non_found_words)
                    acc_rate, sug_rate = get_performance_metrics(tp, fp, tn, fn)
                    print(max_dist_threshold)
                    print(f"acc_rate: {acc_rate:.2f}")
                    print(f"sug_rate: {sug_rate:.2f}")
                    print(tp, fp)
                    print(tn, fn)
                    print()
                    key = key_format.format(max_cost, size_search, max_dist_threshold, no_response_use_sm_priors)
                    with open(os.path.join(calculation_save_folder, f'foundwords_{key}.json'), 'w') as fp:
                        json.dump(found_words, fp)
                    with open(os.path.join(calculation_save_folder, f'nonfoundwords_{key}.json'), 'w') as fp:
                        json.dump(non_found_words, fp)
                    with open(os.path.join(calculation_save_folder, f'wordspredictedbyline_{key}.json'), 'w') as fp:
                        json.dump(words_predicted_by_line, fp)

    if calculate_metrics_roc:
        TPRs, FPRs, acc_rates, sug_rates  = [], [], [], []
        metrics_roc = {
            "params": [],
            "TPRs": [],
            "FPRs": [],
            "acc_rates": [],
            "sug_rates": [],
            "roc": []
        }
        for max_cost in max_costs:
            for size_search in size_searchs:
                for max_dist_threshold in max_dist_thresholds:
                    key = key_format.format(max_cost, size_search, max_dist_threshold, no_response_use_sm_priors)
                    found_words, non_found_words, words_predicted_by_line = load_words(calculation_save_folder, key)
                    # Metrics
                    tp, fp, tn, fn, info_conf = get_confusion_matrix(found_words, non_found_words)
                    acc_rate, sug_rate = get_performance_metrics(tp, fp, tn, fn)
                    metrics_roc["params"].append((max_cost, size_search, max_dist_threshold))
                    metrics_roc["acc_rates"].append(acc_rate)
                    metrics_roc["sug_rates"].append(sug_rate)
                    metrics_roc["roc"].append((tp, fp, tn, fn))
                    print(max_cost, size_search, max_dist_threshold)
                    print(f"acc_rate: {acc_rate:.2f}")
                    print(f"sug_rate: {sug_rate:.2f}")
                    print()
                    words_sep = calculate_metrics_sep(found_words, non_found_words)
                    words_sep["params"] = (max_cost, size_search, max_dist_threshold)
                    np.savez(os.path.join(segmentation_save_folder, f"words_sep_{key}.npz"), **words_sep)
        np.savez(all_metrics_filename, **metrics_roc)