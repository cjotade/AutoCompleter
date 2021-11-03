import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from fast_autocomplete import AutoComplete

from utils import get_keys_startswith_dict, get_lengths_found

class ModelManager:
    def __init__(self, model, count_words, spanish_count_words=None):
        self.model = model
        self.count_words = count_words
        self.spanish_count_words = spanish_count_words
        self.spanish_model = None
        if spanish_count_words:
            self.spanish_model = AutoComplete(words=dict.fromkeys(spanish_count_words.keys(), {}))
        
    def run(self, data, max_cost=1000, size_search=1000, max_dist_threshold=None, scale_mode=None, no_response_use_sm_priors=False):
        """
        Run and get metrics
        """
        found_words, non_found_words, words_predicted_by_line = self.fill_metrics(data, 
                                                                    max_cost=max_cost, 
                                                                    size_search=size_search, 
                                                                    max_dist_threshold=max_dist_threshold, 
                                                                    scale_mode=scale_mode,
                                                                    no_response_use_sm_priors=no_response_use_sm_priors
                                                                    )
        return found_words, non_found_words, words_predicted_by_line

    def predict(self, incomplete_word, max_cost=1000, size_search=1000, max_dist_threshold=None, scale_mode=None, no_response_use_sm_priors=False, **kwargs):
        """
        Predict using models
        """
        # Search in model
        search_responses = self.search_in_model(self.model,
                                        incomplete_word, 
                                        max_cost=max_cost, 
                                        size=size_search
                                        )
        # Filter model response
        filtered_responses = self.filter_response(self.count_words, incomplete_word, search_responses)
        if filtered_responses:
            # Scale words
            scaled_words = scale_count_words(filtered_responses, mode=scale_mode)
            # Use big corpus model for rescale distribution via Bayes posterior
            if self.spanish_model is not None:
                scaled_words = self.scale_with_spanish_model(incomplete_word, scaled_words=scaled_words, max_cost=max_cost, size_search=size_search, scale_mode=scale_mode)
            return self.make_decision_from_dist(scaled_words, max_dist_threshold=max_dist_threshold)
        else:
            # Try to use spanish model priors
            if no_response_use_sm_priors:
                scaled_words = self.scale_with_spanish_model(incomplete_word, max_cost=max_cost, size_search=size_search, scale_mode=scale_mode)
                if scaled_words:
                    return self.make_decision_from_dist(scaled_words, max_dist_threshold=max_dist_threshold)
                else:
                    return None
            else:
                return None

    def search_in_model(self, model, incomplete_word, max_cost=3, size=3):
        """
        Search in the model and get responses
        """
        response = model.search(word=incomplete_word, max_cost=max_cost, size=size)
        response = np.concatenate(response).tolist() if response else np.array(response).tolist()
        return response

    def filter_response(self, count_words, incomplete_word, responses):
        """
        Filter the response by checking:
        Check if the length between res and incomplete_word are not less than 1.
        """
        filtered_responses = {}      
        for res in responses:
            if self._check_len_diff_words(res, incomplete_word):
                cnt_res = count_words.get(res)
                if cnt_res:
                    filtered_responses[res] = cnt_res
        if filtered_responses:
            return filtered_responses
        else:
            return None

    def scale_with_spanish_model(self, incomplete_word, scaled_words=None, max_cost=1000, size_search=1000, scale_mode=None):
        """
        scaled_words are the words obtained from the subject pretrained model. If is not None, then use posterior probabilities, else use spanish_model prior.
        """
        if self.spanish_model:
            # Search in big corpus model
            sm_responses = self.search_in_model(self.spanish_model, incomplete_word, max_cost=max_cost, size=size_search)
            # Filter response from big corpus model
            sm_filtered_responses = self.filter_response(self.spanish_count_words, incomplete_word, sm_responses)
            if sm_filtered_responses:
                sm_scaled_words = scale_count_words(sm_filtered_responses, mode=scale_mode)
                # CASE: Calculate posterior using scaled responses from subject pretrained model
                if scaled_words:
                    posterior_words = {}
                    for k_word, dist_word in scaled_words.items():
                        posterior_words[k_word] = dist_word * sm_scaled_words.get(k_word, 0)
                    scaled_words = posterior_words.copy()
                # CASE: Use only spanish model prior
                else:
                    scaled_words = sm_scaled_words.copy()
                return scaled_words
            else:
                # CASE: No response in spanish model but there are scaled_words
                if scaled_words:
                    for k_word, dist_word in scaled_words.items():
                        scaled_words[k_word] = 0 #? Note here we force a prior 0 (the most conservative model)
                    return scaled_words
                # CASE: No response in spanish model nor scaled_words
                else:
                    return None
        else:
            raise ValueError("Spanish Model is None")

    def make_decision_from_dist(self, scaled_words, max_dist_threshold=None):
        # Decision
        responses, dist_responses = list(scaled_words.keys()), list(scaled_words.values())
        best_response = responses[np.argmax(dist_responses)]
        if max_dist_threshold:
            if np.max(dist_responses) < max_dist_threshold:
                best_response = None
        return best_response, responses, dist_responses

    def fill_metrics(self, data, max_cost=3, size_search=3, max_dist_threshold=None, scale_mode=None, no_response_use_sm_priors=False):
        """
        Iterate over data and get metrics results
        """
        found_words, non_found_words, words_predicted_by_line_arr = [], [], []
        for line in data:
            words_predicted_by_line = 0
            for word in line.split(" "):
                word_is_found = False
                for i in range(len(word)-1):
                    incomplete_word = word[:i+1]
                    if len(incomplete_word) <= 1:
                        continue
                    # Prediction
                    predictions = self.predict(incomplete_word, max_cost=max_cost, size_search=size_search, max_dist_threshold=max_dist_threshold, scale_mode=scale_mode, no_response_use_sm_priors=no_response_use_sm_priors)
                    if predictions:
                        best_response, responses, dist_responses = predictions
                        if best_response:
                            found_words.append({
                                "incomplete_word": incomplete_word,
                                "word": word,
                                "best_response": best_response,
                                "responses": responses,
                                "dist_responses": dist_responses,
                                "correct_decision": best_response == word,
                                "word_in_responses": word in responses
                            })
                            if best_response == word:
                                word_is_found = True
                        else:
                            non_found_words.append({
                                "incomplete_word": incomplete_word,
                                "word": word
                            }) 
                    else:
                        non_found_words.append({
                            "incomplete_word": incomplete_word,
                            "word": word
                        })
                if word_is_found:
                    words_predicted_by_line += 1
            words_predicted_by_line_arr.append(words_predicted_by_line)
        return found_words, non_found_words, words_predicted_by_line_arr
    
    def _check_len_diff_words(self, word1, word2, threshold=1):
        return abs(len(word1) - len(word2)) > threshold
        
    

#########################
# Complementary methods #
#########################

def scale_count_words(cnt_words, mode="empirical"):
    """
    Normalize (or Standarize) frequency of words.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    cnt_words_values = np.array(list(cnt_words.values()))
    if mode == "std" or mode == "min_max":
        scaler = StandardScaler() if mode == "std" else MinMaxScaler()
        scaled_count = scaler.fit_transform(
            np.expand_dims(cnt_words_values, 1)
        )
    elif mode == "empirical":
        scaled_count = cnt_words_values / np.sum(cnt_words_values)
    else:
        return cnt_words
    dist_words = {}
    for i, (word, value) in enumerate(cnt_words.items()):
        dist_words[word] = scaled_count[i].item()
    return dist_words

def get_confusion_matrix(found_words, non_found_words):
    tp, fp, tn, fn = 0, 0, 0, 0
    info = {"tp": [], "fp": [], "tn": [], "fn": []}
    for found_word in found_words:
        correct_decision = found_word["correct_decision"]
        word_in_responses = found_word["word_in_responses"] 
        info_response = {
            "incomplete_word": found_word["incomplete_word"],
            "word": found_word["word"],
            "best_response": found_word["best_response"],
            "responses": found_word["responses"],
            "dist_responses": found_word["dist_responses"]
        }
        if correct_decision and word_in_responses:
            tp += 1
            info["tp"].append(info_response)
        if not correct_decision and word_in_responses:
            fp += 1
            info["fp"].append(info_response)
        if not correct_decision and not word_in_responses:
            fn += 1
            info["fn"].append(info_response)
    tn = len(non_found_words)
    for non_found_word in non_found_words:
        info["tn"].append({
                "incomplete_word": non_found_word["incomplete_word"],
                "word": non_found_word["word"]
            })
    return tp, fp, tn, fn, info

def plot_confusion_matrix(tp, fp, tn, fn, return_norm=True):
    if return_norm:
        sum_res = tp + fp + tn + fn
        tp /= sum_res
        fp /= sum_res
        tn /= sum_res
        fn /= sum_res
    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap([[tp, fp], [tn, fn]], 
                xticklabels=["True", "False"],
                yticklabels=["Positive", "Negative"],
                cmap="Blues", 
                annot=True, 
                fmt="g",
                annot_kws={"fontsize":16},
                ax=ax)
    ax.set_title("Correct decision vs Word in Responses", fontsize=16)
    ax.tick_params(labelsize=14)
    plt.show()

def get_performance_metrics(tp, fp, tn, fn):
    if (tp + fp + fn) != 0:
        acc_rate = tp / (tp + fp + fn)
    else:
        acc_rate = 0
    if (tp + tn + fp + fn) != 0:
        suggestion_rate = (tp + fp + fn) / (tp + tn + fp + fn)
    else:
        suggestion_rate = 0
    return acc_rate, suggestion_rate
    
def calculate_metrics_sep(found_words, non_found_words):
    """
    Calculate metrics by separating the dictionaries by length of words
    """
    # Split by pred words
    found_words_select = get_lengths_found(found_words)
    non_found_words_select = get_lengths_found(non_found_words)
    # Storage words separated by len and params
    words_sep = {
        "len_sel_words": [],
        "TPRs": [],
        "FPRs": [],
        "acc_rates": [],
        "sug_rates": [],
        "roc": []
    }
    for len_sel_words in np.unique(np.array(list(found_words_select.keys()) + list(non_found_words_select.keys()))):
        # Get found words selected by length, return empty list if no len found
        fw_sel = found_words_select.get(len_sel_words, [])
        nfw_sel = non_found_words_select.get(len_sel_words, [])
        # Get confusion matrix
        tp_s, fp_s, tn_s, fn_s, _ = get_confusion_matrix(fw_sel, nfw_sel)
        # performance metrics
        #p_s, r_s, f1_s, acc_s, fpr_s, acc_r_s, sug_r_s = get_performance_metrics(tp_s, fp_s, tn_s, fn_s)
        acc_r_s, sug_r_s = get_performance_metrics(tp_s, fp_s, tn_s, fn_s)
        # Store data
        words_sep["len_sel_words"].append(len_sel_words)
        #words_sep["TPRs"].append(r_s)
        #words_sep["FPRs"].append(fpr_s)
        words_sep["acc_rates"].append(acc_r_s)
        words_sep["sug_rates"].append(sug_r_s)
        words_sep["roc"].append((tp_s, fp_s, tn_s, fn_s))
    return words_sep

def calculate_info_stats(info_conf):
    """
    Calculate statics from confusion matrix
    """
    info_conf_stats = {}
    for cnf in ["tp", "fp", "tn", "fn"]:
        info_stats = {
            "len_incomplete_word": [],
            "len_word": [],
            "len_best_response": []
        }
        for stat_conf in info_conf[cnf]:
            info_stats["len_incomplete_word"].append(len(stat_conf.get("incomplete_word", "")))
            info_stats["len_word"].append(len(stat_conf.get("word", "")))
            info_stats["len_best_response"].append(len(stat_conf.get("best_response", "")))

        info_conf_stats[cnf] = info_stats
    return info_conf_stats