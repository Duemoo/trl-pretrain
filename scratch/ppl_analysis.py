import pickle as pkl
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
import statistics
import numpy as np
from scipy.stats import pearsonr
from scipy import fftpack
import fathon
from fathon import fathonUtils as fu
import powerlaw
import pandas as pd
import seaborn as sns


def fraction_larger_than_first(lst):
    if len(lst) <= 1:
        return 0  # Avoid division by zero if the list has one or no elements
    
    first_value = lst[0]
    count_larger = sum(1 for x in lst[1:] if x > first_value)
    
    return count_larger / (len(lst) - 1)


def check_success(ppls, threshold=0.2, debug=False):
    # print(ppls)
    # print((ppls[0]-min(ppls[:50])), '\t', abs(max(ppls[-50:])-min(ppls[-50:])))
    # assert False
    if (ppls[0]-min(ppls[:25])) > 1.5*(abs(max(ppls[-25:])-min(ppls[-25:]))):
        # if fraction_larger_than_first(ppls)<threshold:
        return True
    else:
        if debug:
            if ppls[0]>ppls[1]:
                print("Fail due to the initial ppl increase")
            if fraction_larger_than_first(ppls)<threshold:
                print("Fail due to the threshold condition")
        return False


def filter_data(ppl_data, sim):
    # Remove outliers based on IQR for ppl
    Q1, Q3 = np.percentile(ppl_data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_indices = [i for i, value in enumerate(ppl_data) if lower_bound <= value <= upper_bound]
    sim_filtered = [sim[i] for i in filtered_indices]
    ppl_filtered = [ppl_data[i] for i in filtered_indices]
    return sim_filtered, ppl_filtered


def draw_violin_contain(contain, hard_contain, ppl, ppl_hard, interval, exp_name):


    assert len(contain)==len(ppl) and len(hard_contain)==len(ppl_hard)

    contain_combined = contain + hard_contain
    contain_group = []
    for c in contain_combined:
        if c:
            contain_group.append('Contianed')
        else:
            contain_group.append('Not contained')
    ppl_combined = ppl + ppl_hard

    df = pd.DataFrame({'category': contain_group, 'value': ppl_combined})

# 
    # fig, axes = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1]})
# 
    # Violin plot
    sns.violinplot(x='category', y='value', data=df)
    # axes[0, 0].set_xlabel('Bins of sim (intervals)')
    # axes[0, 0].set_ylabel('ppl values')
    # axes[0, 0].tick_params(axis='x', rotation=45)

    # # Histogram
    # sns.histplot(ax=axes[1, i], x=sim_filtered, bins=bins, kde=False)
    # axes[1, 0].set_ylabel('Count')
    # axes[1, 0].set_xticks(bins)
    # axes[1, 0].tick_params(axis='x', rotation=45)

    # plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(f"violin/{exp_name}", exist_ok=True)
    filename = f"violin/{exp_name}/{interval}_contain.png"
    plt.savefig(filename)


def draw_violin(sim_dict, ppl, ppl_success, hard, interval, exp_name):
    sim_jaccard = sim_dict["jaccard"]
        
    sim_all_filtered, ppl_all_filtered = filter_data(ppl, sim_jaccard)
    sim_success_filtered, ppl_success_filtered = filter_data(ppl_success, sim_jaccard)

    # Setup the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 16), gridspec_kw={'height_ratios': [3, 1]})

    for i, (sim_filtered, ppl_filtered) in enumerate([(sim_all_filtered, ppl_all_filtered), (sim_success_filtered, ppl_success_filtered)]):
        # Binning the data
        bins = np.linspace(min(sim_filtered), max(sim_filtered), num=9)
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
        sim_binned = np.digitize(sim_filtered, bins, right=True)
        sim_binned = [min(i, len(bin_labels)) for i in sim_binned]

        # Creating a DataFrame for plotting
        data = pd.DataFrame({
            'Group': [bin_labels[i-1] for i in sim_binned],
            'Value': ppl_filtered
        })

        data.sort_values('Group', inplace=True)
        
        # Violin plot
        sns.violinplot(ax=axes[0, i], x='Group', y='Value', data=data)
        # axes[0, i].set_title(f'Violin plo with outliers removed')
        axes[0, i].set_xlabel('Bins of sim (intervals)')
        axes[0, i].set_ylabel('ppl values')
        axes[0, i].tick_params(axis='x', rotation=45)

        # Histogram
        sns.histplot(ax=axes[1, i], x=sim_filtered, bins=bins, kde=False)
        # axes[1, i].set_title(f'Histogram of {label}')
        # axes[1, i].set_xlabel(label)
        axes[1, i].set_ylabel('Count')
        axes[1, i].set_xticks(bins)
        axes[1, i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(f"violin/{exp_name}", exist_ok=True)
    filename = f"violin/{exp_name}/{interval}_{'hard' if hard else 'easy'}.png"
    plt.savefig(filename)


def round(num):
    if num%10<5:
        return num//10*10-1
    else:
        return num//10*10+10-1


def mean_of_arrays(arrays):
    """
    Compute the mean of several 1D numpy arrays.

    :param arrays: List of 1D numpy arrays, all of the same length.
    :return: A 1D numpy array which is the mean of the input arrays.
    """
    stacked_arrays = np.stack(arrays)
    mean_array = np.mean(stacked_arrays, axis=0)
    return mean_array


def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]


def spectrum_analysis(values):
    """
    Perform linear detrending and Fourier analysis on a time-series data.

    :param values: List of floats representing the time-series data.
    :return: Plot of the frequency spectrum.
    """

    # Time parameters (assuming equal spacing)
    N = len(values)  # Number of data points
    T = 1.0 / N  # Assuming unit time interval between data points

    # Linear Detrending
    times = np.arange(N)
    detrended = values - np.poly1d(np.polyfit(times, values, 1))(times)

    # Fourier Transform
    freq_values = fftpack.fft(detrended)
    freqs = fftpack.fftfreq(N, T)
    freq_magnitudes = np.abs(freq_values) * 1 / N

    # Normalizing to make the area under the curve 1
    total_area = np.sum(freq_magnitudes) * (freqs[1] - freqs[0])  # Approximate the integral
    normalized_magnitudes = freq_magnitudes / total_area
    
    # Plotting the Frequency Spectrum
    # plt.figure(figsize=(10, 5))
    # plt.plot(freqs[:N // 2][1:], normalized_magnitudes[:N // 2][1:])  # Plot only the positive frequencies
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.title('Frequency Spectrum')
    # plt.grid(True)
    # plt.show()
    # plt.savefig('spectrum_mem.png')
    return freqs[:N // 2][1:], normalized_magnitudes[:N // 2][1:]


def remove_outliers_iqr(data, multiplier=1.5, log=False):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    if log:
        print(f"{len(data)-len(filtered_data)}/{len(data)} datapoints removed")
    return filtered_data


def load_json(path):
    with open(path) as f:
        data = [json.loads(l.strip()) for l in f]
        # data = json.load(f)
    return data


def mean(l):
    return sum(l)/len(l)


def sort_idx(scores):
        sorted_pairs = sorted(zip(scores, lst), key=lambda x: x[1], reverse=True)
        return [index for index, value in sorted_pairs]


def get_perturb_indices(l, max_len=500, margin=25):
    if len(l)==0:
        return []
    else:
        result = []
        for i in range(len(l)-1):
            if l[i]+margin<l[i+1]:
                result.extend(list(range(l[i]+margin,l[i+1])))
        if l[-1]<max_len-margin:
            result.extend(list(range(l[-1]+margin,max_len)))

        return result


def fit_powerlaw(raw_data, mode):
    # Fit data to a power-law distribution
    data = [abs(d) for d in raw_data]
    fit = powerlaw.Fit(data)
    alpha = fit.alpha
    xmin = fit.xmin

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # First subplot: Power-law PDF
    fit.plot_pdf(color='b', linestyle='-', ax=axs[0])
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=axs[0])

    # Compare power-law fit to an exponential distribution
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f'{mode} - Likelihood Ratio: {R}, p-value: {p}')

    axs[0].set_title(f'Power-law fit: α={alpha:.2f}, xmin={xmin}')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Probability Density Function')
    axs[0].text(0.6, 0.95, f'p-value: {p:.6f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

    # Second subplot: Histogram of data with mean and standard deviation
    mean_data = np.mean(data)
    std_data = np.std(data)
    median_data = np.median(data)
    bins = np.linspace(0, 2, 21)
    axs[1].hist(data, bins=bins, edgecolor='black')
    axs[1].set_title('Histogram of Data')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    # Display mean and standard deviation
    axs[1].text(0.6, 0.9, f'Mean: {mean_data:.2f}\nMedian: {median_data:.2f}\nStd: {std_data:.2f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

    # Third subplot: Histogram of raw data with mean and standard deviation
    mean_raw_data = np.mean(raw_data)
    std_raw_data = np.std(raw_data)
    median_raw_data = np.median(raw_data)
    bins = np.linspace(-2, 2, 41)
    axs[2].hist(raw_data, bins=bins, edgecolor='black')
    axs[2].set_title('Histogram of Raw Data')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')
    # Display mean and standard deviation
    axs[2].text(0.6, 0.9, f'Mean: {mean_raw_data:.2f}\nMedian: {median_raw_data:.2f}\nStd: {std_raw_data:.2f}', transform=axs[2].transAxes, fontsize=12, verticalalignment='top')


    # Save the figure
    plt.savefig(f'powerlaw/{args.exp_name[0]}_{mode}.png')
    np.save(f'powerlaw/raw/{args.exp_name[0]}_{mode}.npy')

    # Show plot
    plt.show()


def measure_scores(result, train_indices, premem=False, interval=10000):

    with open('similarity.json', 'r') as f:
        sim_data = json.load(f)
    with open('fictional_knowledge_contain.json', 'r') as f:
        contain_data = json.load(f)

    probe_ppls = [instance["ppl_probe"] if len(instance["ppl_probe"])>0 else [[0.0 for i in range(12)] for i in range(156)] for instance in result]
    probe_ppls = list(map(list, zip(*probe_ppls)))
    train_ppls = [instance["ppl_train"] if len(instance["ppl_train"])>0 else [0.0 for i in range(156)] for instance in result]
    train_ppls = list(map(list, zip(*train_ppls)))
    margin=50

    memorizability = []
    generalizability = []
    mem_freq_per_ex = []
    gen_freq_per_ex = []
    mem_learnability_per_ex = []
    gen_learnability_per_ex = []
    gen_learnability_easy_per_ex = []
    gen_learnability_hard_per_ex = []
    gen_success_learnability_easy_per_ex = []
    gen_success_learnability_hard_per_ex = []
    gen_fluc_per_ex = []
    mem_fluc_per_ex = []
    similarity_easy_per_ex = {"jaccard": [], "rouge_l": []}
    similarity_hard_per_ex = {"jaccard": [], "rouge_l": []}
    freq = None
    success_count_easy=0
    success_count_hard=0
    contain = []
    hard_contain = []

    for ex_idx in tqdm(range(len(probe_ppls))):

        train_idx = train_indices[ex_idx] if not premem else []
        n_probes = len(probe_ppls[ex_idx][0])        

        # before_encounter_indices = list(range(1,train_idx[0])) if len(train_idx)>0 else list(range(1, 500))
        # perturb_indices = get_perturb_indices(train_idx)
        if len(train_idx)!=0 and not premem:
            similarity_easy_per_ex["jaccard"].extend(sim_data['normal']['jaccard'][ex_idx*5:ex_idx*5+5])
            similarity_easy_per_ex["rouge_l"].extend(sim_data['normal']['rouge_l'][ex_idx*5:ex_idx*5+5])
            similarity_hard_per_ex["jaccard"].extend(sim_data['hard']['jaccard'][ex_idx*5:ex_idx*5+5])
            similarity_hard_per_ex["rouge_l"].extend(sim_data['hard']['rouge_l'][ex_idx*5:ex_idx*5+5])

        for j in range(n_probes):
            ppls = [d[j] for d in probe_ppls[ex_idx]]
            
            if len(train_idx)!=0 and not premem:
                values=ppls[train_idx[-1]:train_idx[-1]+margin]
                sp=min(range(len(values)), key=values.__getitem__)+train_idx[-1]
                # min_ppl=min(ppls[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(ppls[sp-10:sp+10])
                # init_ppl=ppls[train_idx[-1]-1]
                init_ppl=ppls[train_idx[-1]-2]

                generalizability.append((1-min_ppl/init_ppl)*100)

                # Freq analysis
                freq_x, freq_y = spectrum_analysis(ppls[sp:sp+400])
                freq = freq_x 
                if interval>0:
                    last_ppl=ppls[round(sp+interval)]
                else:
                    last_ppl=ppls[sp+interval]
                gen_freq_per_ex.append(freq_y)
                gen_learnability_per_ex.append((1-last_ppl/init_ppl))

                values_with_prev = ppls[train_idx[-1]-2:train_idx[-1]+margin]
                if j<5:
                    gen_learnability_easy_per_ex.append((1-last_ppl/init_ppl))
                    contain.append(contain_data[ex_idx]["contain"][j])
                    if check_success(values_with_prev):
                        success_count_easy += 1
                        gen_success_learnability_easy_per_ex.append((1-last_ppl/init_ppl))
                else:
                    gen_learnability_hard_per_ex.append((1-last_ppl/init_ppl))
                    hard_contain.append(contain_data[ex_idx]["hard_contain"][j-5])
                    if check_success(values_with_prev):
                        success_count_hard += 1
                        gen_success_learnability_hard_per_ex.append((1-last_ppl/init_ppl))
                # gen_fluc_per_ex.append(1-last_ppl/min_ppl)
                # segment = ppls[sp:sp+400]
                # gen_fluc = calculate_fluc(segment)
                # gen_fluc_per_ex.append(gen_fluc)
                gen_fluc_per_ex.append((last_ppl-min_ppl)/abs(init_ppl-min_ppl))
                # pre_gen_fluc_per_ex.append(1-ppls[99]/ppls[0])


            elif premem:
                freq_x, freq_y = spectrum_analysis(ppls[100:500])
                freq = freq_x 
                freq_x, freq_y = spectrum_analysis(ppls[100:500])
                freq = freq_x 
                gen_freq_per_ex.append(freq_y)

                values=ppls[500:500+margin]
                sp=min(range(len(values)), key=values.__getitem__)+500
                # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(ppls[sp-10:sp+10])
                
                gen_fluc_per_ex.append((ppls[round(500+interval)]-min_ppl)/abs(ppls[500]-min_ppl))


        if len(train_idx)!=0 and not premem:
            train_ppl = train_ppls[ex_idx]
            # if n_probes>0:
            if True:
                values=train_ppl[train_idx[-1]:train_idx[-1]+margin]
                sp=min(range(len(values)), key=values.__getitem__)+train_idx[-1]
                # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(train_ppl[sp-10:sp+10])

                # last_ppl=train_ppl[-1]
                init_ppl=train_ppl[train_idx[-1]-1]

                memorizability.append((1-min_ppl/init_ppl)*100)

                # Frequency analysis
                _, freq_y = spectrum_analysis(train_ppl[sp:sp+400])
                mem_freq_per_ex.append(freq_y)
                if interval>0:
                    last_ppl=train_ppl[round(sp+interval)]
                else:
                    last_ppl=train_ppl[sp+interval]
                mem_learnability_per_ex.append((1-last_ppl/init_ppl))
                # if mem_learnability < 5:
                #     pass
                # segment = train_ppl[sp:sp+400]
                # mem_fluc = calculate_fluc(segment)
                # mem_fluc_per_ex.append(mem_fluc)
                mem_fluc_per_ex.append((last_ppl-min_ppl)/abs(init_ppl-min_ppl))
                # pre_mem_fluc_per_ex.append(1-train_ppl[99]/train_ppl[0])
                # print(f"last ppl: {last_ppl} / min ppl: {min_ppl} / init ppl: {init_ppl}")
        
            else:
                pass
                # train_ppl = train_ppls[ex_idx]
                # pre_mem_fluc_per_ex.append(abs(1-train_ppl[400]/train_ppl[0]))

        elif premem:
            train_ppl = train_ppls[ex_idx]
            freq_x, freq_y = spectrum_analysis(train_ppl[100:500])
            freq = freq_x 
            freq_x, freq_y = spectrum_analysis(train_ppl[100:500])
            freq = freq_x 
            mem_freq_per_ex.append(freq_y)

            values=train_ppl[500:500+margin]
            sp=min(range(len(values)), key=values.__getitem__)+500
            # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
            min_ppl=mean(train_ppl[sp-10:sp+10])

            mem_fluc_per_ex.append((train_ppl[round(500+interval)]-min_ppl)/abs(train_ppl[500]-min_ppl))

    # draw_violin(similarity_easy_per_ex, gen_learnability_easy_per_ex, gen_success_learnability_easy_per_ex, hard=False, interval=args.interval, exp_name=args.exp_name[0][:-5])
    # draw_violin(similarity_hard_per_ex, gen_learnability_hard_per_ex, gen_success_learnability_hard_per_ex, hard=True, interval=args.interval, exp_name=args.exp_name[0][:-5])
    draw_violin_contain(contain, hard_contain, gen_learnability_easy_per_ex, gen_learnability_hard_per_ex, interval=args.interval, exp_name=args.exp_name[0][:-5])

    # remove outliers
    if not premem:
        # print(memorizability)
        # print(generalizability)
        memorizability = remove_outliers_iqr(memorizability)
        # train_volatility_per_ex = remove_outliers_iqr(train_volatility_per_ex)
        if len(generalizability)>0:
            generalizability = remove_outliers_iqr(generalizability)
        # volatility_per_ex = remove_outliers_iqr(volatility_per_ex)
        mem_learnability_per_ex = remove_outliers_iqr(mem_learnability_per_ex, log=True)
        mem_fluc_per_ex = remove_outliers_iqr(mem_fluc_per_ex)
        # if len(gen_learnability_per_ex)>0:
        orig_len = len(gen_learnability_easy_per_ex)
        gen_learnability_per_ex = remove_outliers_iqr(gen_learnability_per_ex, log=True)
        gen_learnability_easy_per_ex = remove_outliers_iqr(gen_learnability_easy_per_ex, log=True)
        gen_learnability_hard_per_ex = remove_outliers_iqr(gen_learnability_hard_per_ex, log=True)
        gen_success_learnability_easy_per_ex = remove_outliers_iqr(gen_success_learnability_easy_per_ex, log=True)
        gen_success_learnability_hard_per_ex = remove_outliers_iqr(gen_success_learnability_hard_per_ex, log=True)

    # Plot averqge frequency spectrum
    mem_freq = mean_of_arrays(mem_freq_per_ex)
    if len(gen_freq_per_ex)>0:
        gen_freq = mean_of_arrays(gen_freq_per_ex)

    # print(mem_freq, '\n\n\n')
    # print(gen_freq)

    # Frequency plot
    # plt.figure(figsize=(10, 5))
    # visible=25
    # plt.plot(freq[:visible], mem_freq[:visible], label='Memorization')  # Plot only the positive frequencies
    # plt.plot(freq[:visible], gen_freq[:visible], label='Generalization')  # Plot only the positive frequencies
    # plt.xlabel('Frequency')
    # plt.ylabel('Normalized Amplitude')
    # plt.title('Frequency Spectrum')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'spectrums/{args.exp_name[0]}.png')

    # np.save(f'spectrums/raw/{args.exp_name[0]}_mem.npy', mem_freq)
    # np.save(f'spectrums/raw/{args.exp_name[0]}_gen.npy', gen_freq)


    # print(f"mem center of mass: {np.sum(freq * mem_freq) / np.sum(mem_freq)}")
    # print(f"gen center of mass: {np.sum(freq * gen_freq) / np.sum(gen_freq)}")
        

    
    if not premem:
        # print(f"memorizability: mean {mean(memorizability)} / {statistics.pstdev(memorizability)}")
        print(f"mem_learnability: {mean(mem_learnability_per_ex)}")
        # print(f"mem_learnability_stdev: {statistics.pstdev(mem_learnability_per_ex)}")
        # print(f"mem_fluc: {mean(mem_fluc_per_ex)}")
        # print(f"train volatility: mean {mean(train_volatility_per_ex)} / {statistics.pstdev(train_volatility_per_ex)}")
        print()
        # print(f"generalizability: mean {mean(generalizability)} / {statistics.pstdev(generalizability)}")
        # print(f"gen_learnability: {mean(gen_learnability_per_ex)}")
        # print(f"gen_learnability_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        print()
        print(f"gen_learnability_easy: {mean(gen_learnability_easy_per_ex)}")
        # print(f"gen_success_learnability_easy: {mean(gen_success_learnability_easy_per_ex)}")
        # print(f"gen_learnability_easy_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        print()
        print(f"gen_learnability_hard: {mean(gen_learnability_hard_per_ex)}")
        # print(f"gen_success_learnability_hard: {mean(gen_success_learnability_hard_per_ex)}")
        print()
        # print(f"easy success fraction: {success_count_easy/orig_len}")
        # print(f"hard success fraction: {success_count_hard/orig_len}")
        # print(f"gen_learnability_hard_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        # print(f"gen_fluc: {mean(gen_fluc_per_ex)}")
        # print(f"len_notrain: {len(pre_mem_fluc_per_ex)}")
    # print(f"gen volatility: mean {mean(volatility_per_ex)} / {statistics.pstdev(volatility_per_ex)}")


    # Fit the data to a power-law distribution

    # fit_powerlaw(pre_gen_fluc_per_ex, mode='pre_gen')
    # fit_powerlaw(gen_fluc_per_ex, mode='gen')
    # fit_powerlaw(pre_mem_fluc_per_ex, mode='pre-mem')
    # fit_powerlaw(mem_fluc_per_ex, mode='mem')


def plot_ppl_with_trained_at(results, exp_names=['105b', '1T', '2T', '3T'], save_dir='plot', train_indices=None):
    
    steps = [data["step"] for data in results[0]][:10600]
    all_probe_ppls = []
    all_train_ppls = []
    for result in results:
        # probe_ppls = [instance["ppl_probe"] for instance in result]
        probe_ppls = [instance["ppl_probe"] if len(instance["ppl_probe"])>0 else [[0.0 for i in range(12)] for i in range(156)] for instance in result][:10600]
        for i in range(len(probe_ppls)):
            if 0.0 in probe_ppls[i][0]:
                probe_ppls[i] = probe_ppls[i-1]
        all_probe_ppls.append(list(map(list, zip(*probe_ppls))))
        # train_ppls = [instance["ppl_train"] for instance in result]
        train_ppls = [instance["ppl_train"] if len(instance["ppl_train"])>0 else [0.0 for i in range(156)] for instance in result][:10600]
        for i in range(len(train_ppls)):
            if 0.0 in train_ppls[i]:
                train_ppls[i] = train_ppls[i-1]
        all_train_ppls.append(list(map(list, zip(*train_ppls))))

    # plt.figure(figsize=(16, 20))
    for ex_idx in tqdm(range(len(all_probe_ppls[0]))):
        plt.figure(figsize=(32, 15))
        # plt.figure(figsize=(16, 30))

        for result_idx in range(len(results)):
            n_probes = len(all_probe_ppls[result_idx][ex_idx][0])
            # print(n_probes)
            for j in range(n_probes):
                margin=50
                if len(train_indices[ex_idx])>0:
                    last_train_idx = train_indices[ex_idx][-1]
                    values=[all_probe_ppls[result_idx][ex_idx][i-2][j] for i in range(last_train_idx, last_train_idx+50)]
                    # print(values)
                    color = 'blue' if check_success(values) else 'gray'
                else:
                    color = 'green'
                
                plt.subplot(1+n_probes, len(results), result_idx+1+j*len(results))
                plt.plot(steps, [d[j] for d in all_probe_ppls[result_idx][ex_idx]], color=color)
                try:
                    x_vals=[i-101 for i in train_indices[ex_idx]]
                    y_vals=[all_probe_ppls[result_idx][ex_idx][i-1][j] for i in train_indices[ex_idx]]
                    plt.scatter(x_vals, y_vals, color='red', s=24)

                    x_prev_vals=[i-102 for i in train_indices[ex_idx]]
                    y_prev_vals=[all_probe_ppls[result_idx][ex_idx][i-2][j] for i in train_indices[ex_idx]]
                    plt.scatter(x_prev_vals, y_prev_vals, color='black', s=24)
                except:
                    pass

                plt.xlabel('Training Step')
                plt.ylabel(f'Perplexity')
                plt.grid(True)
                
            if True:
                plt.subplot(1+n_probes, len(results), result_idx+1+n_probes*len(results))
                plt.plot(steps, all_train_ppls[result_idx][ex_idx])
                try:
                    x_vals=[i-101 for i in train_indices[ex_idx]]
                    y_vals=[all_train_ppls[result_idx][ex_idx][i-1] for i in train_indices[ex_idx]]
                    plt.scatter(x_vals, y_vals, color='red', s=24)

                    x_prev_vals=[i-102 for i in train_indices[ex_idx]]
                    y_prev_vals=[all_train_ppls[result_idx][ex_idx][i-2] for i in train_indices[ex_idx]]
                    plt.scatter(x_prev_vals, y_prev_vals, color='black', s=24)
                except:
                    pass
                
                plt.xlabel('Training Step')
                plt.ylabel('Perplexity')
                plt.grid(True)

        # Save the figure to a file
        plt.savefig(os.path.join(save_dir, str(ex_idx)+'.png'), bbox_inches='tight')
        plt.close()


def main(args):

    # Filtered samples
    # measure_indices = [
    #         1, 3, 5, 7, 8, 9, 11, 15, 16, 17, \
    #         20, 22, 23, 25, 26, 27, 29, 30, 31, 32, \
    #         34, 37, 39, 40, 41, 43, 44, 45, 46, 47, \
    #         49, 51, 53, 54, 55, 59, 60, 62, 65, 66, \
    #         70, 71, 73, 76, 82, 83, 84, 85, 86, 89, \
    #         91, 92, 97, 98, 103, 106, 110, 112, 113] # 59 examples
    measure_indices = range(156)

    results=[load_json(os.path.join(args.base_dir, 'results/logs/', exp_name)) for exp_name in args.exp_name]
    with open(os.path.join(args.base_dir, 'results/logs/', args.text_log), 'r') as f:
        text_log = json.load(f)


    
    wiki='wiki' in args.exp_name[0]
    
    if wiki:
        dataset_fpath=os.path.join(args.base_dir, 'custom_knowledge/wikipedia_probe.json')
        with open(dataset_fpath, 'r') as f:
            dataset = json.load(f)
            ref_texts = [ref['train_context'] for ref in dataset]
    else:
        with open(os.path.join(args.base_dir, 'custom_knowledge/fictional_knowledge_combined.json'), 'r') as f:
            dataset = json.load(f)
            ref_texts=[d['train_context'] for d in dataset]

    # print(ref_texts)

    train_indices=[[] for i in range(len(ref_texts))]
    for step, texts in tqdm(text_log.items()):
        if int(step)<=100:
            continue
        for text in texts:
            if wiki:
                # index = None
                # for i, ref in enumerate(ref_texts):
                #     # print(text[:128], ref[:128])
                #     if levenshtein(text[:32], ref[:32]) < 5:
                #         index = i
                #         # print(index)
                #         break
                # if index:
                #     train_indices[index].append(int(step))

                # Pre-computed
                train_indices = [[], [442], [335], [238], [383], [], [377], [480], [468], [459], [357], [391], [227], [228], [166], [441], [355], [495], [], [332], [448], [302], [106], [156], [332], [334], [418], [182], [], [324], [], [491], [471], [], [389], [428], [], [346], [265], [222], [407], [], [332], [261], [401], [189], [114], [], [348], [357], [243], [398], [], [], [470], [], [495], [149], [], [314], [308], [], [277], [], [216], [500], [297], [352], [169], [170], [], [370], [202], [244], [232], [240], [148], [364], [439], [], [245], [261], [], [], [475], [208], [224], [476], [353], [], [179], [201], [487], [148], [306], [339], [450], [420], [274], [217], [182], [188], [457], [254], [], [372], [], [213], [474], [171], [], [479], [289], [], [186], [228], [440], [155], [113], [310], [404], [486], [360], [444], [309], [494], [123], [370], [171], [], [], [198], [422], [337], [226], [164], [], [247], [], [217], [404], [], [202], [268], [402], [334], [321], [404], [227], []]
            else:
                with open('precomputed_idx_64.json', 'r') as f:
                    train_indices = json.load(f)
    #             try:
    #                 index = ref_texts.index(text)
    #                 train_indices[index].append(int(step))
    #             except:
    #                 # continue
    #                 pass
    #             index = None
    #             for i, ref in enumerate(ref_texts):
    #                 # print(text[:128], ref[:128])
    #                 if levenshtein(text[:32], ref[:32]) < 10:
    #                     index = i
    #                     # print(index)
    #                     break
    #             if index and index not in train_indices[index]:
    #                 train_indices[index].append(int(step))
    # print(len(train_indices))
    # with open('precomputed_idx_64.json', 'w') as f:
    #     json.dump(train_indices, f, indent=4)
    # assert False
    # assert False

    if args.mode=='draw_figures':
        os.makedirs(args.save_dir, exist_ok=-True)
        plot_indices = range(156,196)
        plot_ppl_with_trained_at(results, save_dir=args.save_dir, train_indices=train_indices)



    
    elif args.mode=='measure_scores':
        # measure_indices = list(range(len(per_ex)))
        measure_scores(results[0], train_indices, interval=args.interval)

    else:
        raise NotImplementedError
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--base_dir', type=str, default="/mnt/nas/hoyeon/trl-pretrain/")
    parser.add_argument('--save_dir', type=str, default="test")
    parser.add_argument('--exp_name', nargs='+', required=True)
    parser.add_argument('--text_log', type=str, required=True)
    parser.add_argument('--mode', type=str, default="draw_figures")
    parser.add_argument('--interval', type=int, default="10000")

    # Parse the arguments
    args = parser.parse_args()

    main(args)