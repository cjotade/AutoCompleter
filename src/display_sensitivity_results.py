import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Calculation params
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

    all_metrics = np.load(all_metrics_filename)
    params = all_metrics["params"]
    acc_rates = all_metrics["acc_rates"]
    sug_rates = all_metrics["sug_rates"]

    for param, sug_r, acc_r in zip(params, sug_rates, acc_rates):
        #if param[-1] in [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #    continue
        plt.plot(sug_r, acc_r, "o", label=f"distr_th: {param[-1]}")
    plt.xlabel("suggestion rates", fontsize=16)
    plt.ylabel("accuracy rates", fontsize=16)
    plt.xlim(-0.02, 1)
    plt.ylim(-0.02, 1.02)
    plt.title(f"max_cost:{max_costs[0]} - size_search:{size_searchs[0]} - no_response_use_sm_priors:{no_response_use_sm_priors}")
    plt.legend()
    plt.grid()
    plt.show()

    for param, sug_r, acc_r in zip(params, sug_rates, acc_rates):
        #if param[-1] in [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #    continue
        plt.plot(sug_r, acc_r, "o", label=f"distr_th: {param[-1]}")
        
    plt.xlabel("suggestion rates", fontsize=16)
    plt.ylabel("accuracy rates", fontsize=16)
    plt.xlim(-0.02, 1)
    plt.ylim(-0.02, 1.02)
    plt.title(f"max_cost:{max_costs[0]} - size_search:{size_searchs[0]} - no_response_use_sm_priors:{no_response_use_sm_priors}")
    plt.legend()
    plt.grid()
    plt.show()

    for max_cost in max_costs:
        for size_search in size_searchs:
            for max_dist_threshold in max_dist_thresholds:
                key = key_format.format(max_cost, size_search, max_dist_threshold, no_response_use_sm_priors)
                words_sep = np.load(os.path.join(segmentation_save_folder, f"words_sep_{key}.npz"))
                len_sel_words = words_sep["len_sel_words"]
                a_rates = words_sep["acc_rates"]
                s_rates = words_sep["sug_rates"]
                for i, (sug_rate, acc_rate) in enumerate(zip(s_rates, a_rates)):
                    plt.scatter(sug_rate, acc_rate, color=f"C{i}", label=f"in_chars:{len_sel_words[i]}")
                    if i > 8:
                        #! Note that here is just plot until i > 8
                        break
                plt.xlabel("suggestion rates", fontsize=16)
                plt.ylabel("accuracy rates", fontsize=16)
                plt.xlim(0, 1)
                plt.ylim(0, 1.02)
                plt.legend()
                plt.grid()
                plt.title(f"max_cost:{max_cost} - size_search:{size_search} - max_dist_thresh:{max_dist_threshold}")
                plt.show()