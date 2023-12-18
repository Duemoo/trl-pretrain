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


def remove_outliers_iqr(data, multiplier=2.0):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    # print(f"{len(data)-len(filtered_data)}")
    return filtered_data


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


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


def measure_scores(result, train_indices):
    
    steps = [data["step"] for data in result]
    probe_ppls = [instance["ppl_probe"] for instance in result]
    probe_ppls = list(map(list, zip(*probe_ppls)))
    train_ppls = [instance["ppl_train"] for instance in result]
    train_ppls = list(map(list, zip(*train_ppls)))

    corr_coeff_per_ex = []
    p_per_ex = []
    pop_corr_coeff_per_ex = []
    pop_p_per_ex = []
    ppl_drop_per_ex = []
    avg_ppl_fluc_before_train_per_ex = []
    avg_ppl_fluc_after_train_per_ex = []
    volatility_per_ex = []
    train_volatility_per_ex = []
    margin=50

    for ex_idx in range(len(probe_ppls)):

        if ex_idx>155:
            break


        train_idx = train_indices[ex_idx]
        n_probes = len(probe_ppls[ex_idx][0])

        corr_coeff = []
        ps = []
        ppl_fluc_after_train = []
        ppl_fluc_before_train = []
        ppl_drop_after_train = []
        volatility = []

        before_encounter_indices = list(range(1,train_idx[0])) if len(train_idx)>0 else list(range(1, 500))
        perturb_indices = get_perturb_indices(train_idx)


        for j in range(n_probes):
            ppls = [d[j] for d in probe_ppls[ex_idx]]
            
            coeff, p = pearsonr(train_ppls[ex_idx], ppls)
            corr_coeff.append(coeff)
            ps.append(p)
            if n_probes>1:
                ppl_fluc_before_train.append(mean([abs((1-ppls[idx]/ppls[idx-1])*100) for idx in before_encounter_indices]))
                if len(perturb_indices)>0:
                    ppl_fluc_after_train.append(mean([abs((1-ppls[idx]/ppls[idx-1])*100) for idx in perturb_indices]))
                if len(train_idx)!=0:
                    min_ppl=min(ppls[train_idx[-1]:train_idx[-1]+margin])
                    # last_ppl=ppls[train_idx[-1]+500]
                    last_ppl=ppls[-1]
                    volatility.append((1-last_ppl/min_ppl)*100)
                    # if len(train_idx)==1:
                    #     if train_idx[0]<400:
                    #         volatility.append(((1-ppls[train_idx[0]+100]/ppls[train_idx[0]+margin])*100))
                    # else:
                    #     if train_idx[0]+100<train_idx[1]:
                    #         volatility.append(((1-ppls[train_idx[0]+100]/ppls[train_idx[0]+margin])*100))

        
        if len(train_idx)!=0:
            train_ppl = train_ppls[ex_idx]
            if n_probes>1:
                min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
                last_ppl=train_ppl[-1]
                train_volatility_per_ex.append((1-last_ppl/min_ppl)*100)
                # if len(train_idx)==1:
                #     if train_idx[0]<400:
                #         train_volatility_per_ex.append(((1-train_ppl[train_idx[0]+100]/train_ppl[train_idx[0]+margin])*100))
                # else:
                #     if train_idx[0]+100<train_idx[1]:
                #         train_volatility_per_ex.append(((1-train_ppl[train_idx[0]+100]/train_ppl[train_idx[0]+margin])*100))


        if n_probes>1:
            avg_ppl_fluc_before_train_per_ex.append(mean(ppl_fluc_before_train))
            if len(perturb_indices)>0:
                avg_ppl_fluc_after_train_per_ex.append(mean(ppl_fluc_after_train))
            if len(volatility)>0:
                volatility_per_ex.append(mean(volatility))


        if n_probes>1:
            corr_coeff_per_ex.append(mean(corr_coeff))
            p_per_ex.append(mean(ps))
        else:
            pop_corr_coeff_per_ex.append(mean(corr_coeff))
            pop_p_per_ex.append(mean(ps))

    print(len(volatility_per_ex), mean(volatility_per_ex))
    print(mean(train_volatility_per_ex))

    # print(mean(corr_coeff_per_ex), mean(p_per_ex))
    # print(mean(pop_corr_coeff_per_ex), mean(pop_p_per_ex))
    # print(sorted(range(len(corr_coeff_per_ex)), key=lambda i: corr_coeff_per_ex[i])[:10])
    # print(sorted(range(len(corr_coeff_per_ex)), key=lambda i: corr_coeff_per_ex[i])[-10:])
    print(mean(avg_ppl_fluc_before_train_per_ex), mean(avg_ppl_fluc_after_train_per_ex))


def plot_ppl_with_trained_at(results, exp_names=['105b', '1T', '1.5T', '2T'], save_dir='plot', train_indices=None):
    
    steps = [data["step"] for data in results[0]]
    all_probe_ppls = []
    all_train_ppls = []
    for result in results:
        probe_ppls = [instance["ppl_probe"] for instance in result]
        all_probe_ppls.append(list(map(list, zip(*probe_ppls))))
        train_ppls = [instance["ppl_train"] for instance in result]
        all_train_ppls.append(list(map(list, zip(*train_ppls))))

    # plt.figure(figsize=(16, 20))
    for ex_idx in tqdm(range(len(all_probe_ppls[0]))):
        plt.figure(figsize=(50, 20))

        for result_idx in range(len(results)):
            n_probes = len(all_probe_ppls[result_idx][ex_idx][0])
            # print(n_probes)
            for j in range(n_probes):
                plt.subplot(1+n_probes, len(results), result_idx+1+j*len(results))
                # print(f"steps: {len(steps)}\n\nppls: {len()}")
                plt.plot(steps, [d[j] for d in all_probe_ppls[result_idx][ex_idx]], label='ppl values')
                try:
                    x_vals=[i-101 for i in train_indices[ex_idx]]
                    y_vals=[all_probe_ppls[result_idx][ex_idx][i-1][j] for i in train_indices[ex_idx]]
                    plt.scatter(x_vals, y_vals, color='red', s=12)

                    x_prev_vals=[i-102 for i in train_indices[ex_idx]]
                    y_prev_vals=[all_probe_ppls[result_idx][ex_idx][i-2][j] for i in train_indices[ex_idx]]
                    plt.scatter(x_prev_vals, y_prev_vals, color='black', s=12)
                except:
                    pass

                plt.xlabel('Training Step')
                plt.ylabel(f'probe{j}_ppl Value')
                plt.title(f'{exp_names[result_idx]}_probe{j}')
                plt.legend()
                plt.grid(True)
                

            plt.subplot(1+n_probes, len(results), result_idx+1+n_probes*len(results))
            plt.plot(steps, all_train_ppls[result_idx][ex_idx], label='ppl values')
            try:
                x_vals=[i-101 for i in train_indices[ex_idx]]
                y_vals=[all_train_ppls[result_idx][ex_idx][i-1] for i in train_indices[ex_idx]]
                plt.scatter(x_vals, y_vals, color='red', s=12)

                x_prev_vals=[i-102 for i in train_indices[ex_idx]]
                y_prev_vals=[all_train_ppls[result_idx][ex_idx][i-2] for i in train_indices[ex_idx]]
                plt.scatter(x_prev_vals, y_prev_vals, color='black', s=12)
            except:
                pass
            
            plt.xlabel('Training Step')
            plt.ylabel('train_ppl Value')
            plt.title(f'{exp_names[result_idx]}_train')
            plt.legend()
            plt.grid(True)

        # Save the figure to a file
        plt.savefig(os.path.join(save_dir, str(ex_idx)+'.png'), bbox_inches='tight')
        plt.close()


def measure_ppl_drop(per_exs, measure_indices, exclude_pop, remove_outliers=True):
    assert len(per_exs)==1
    per_ex = per_exs[0]
    avg_ppl_drop_per_ex = []
    avg_ppl_fluc_stdev_per_ex = []
    avg_ppl_fluc_abs_per_ex = []
    overall_ppl_drop_per_ex = []
    forget_ratio_per_ex = []
    
    for idx, data in enumerate(per_ex.items()):
        if 'pop' in data[0] and exclude_pop:
            continue
        if idx in measure_indices:
            ppl_drop_on_train = []
            ppl_fluc_not_on_train = []
            forget_ratio = []
            train_idx = data[1]['trained_at']
            ppl = data[1]['ppl']
            cache = None

            for step in range(len(ppl)):
                if step in train_idx:
                    if step!=0:
                        ppl_drop_on_train.append((1-ppl[step]/ppl[step-1])*100)
                        if cache:
                            forget_ratio.append((ppl[step-1]/cache-1)*100)
                        cache = ppl[step]
                        
                else:
                    if step!=0:
                        ppl_fluc_not_on_train.append((1-ppl[step]/ppl[step-1])*100)
            
            # ppl_drop_on_train = remove_outliers_iqr(ppl_drop_on_train)
            if remove_outliers:
                ppl_fluc_not_on_train = remove_outliers_iqr(ppl_fluc_not_on_train)
                ppl_drop_on_train = remove_outliers_iqr(ppl_drop_on_train)
                forget_ratio = remove_outliers_iqr(forget_ratio)
                
            avg_ppl_drop_per_ex.append(sum(ppl_drop_on_train)/len(ppl_drop_on_train))
            avg_ppl_fluc_stdev_per_ex.append(statistics.pstdev(ppl_fluc_not_on_train))
            avg_ppl_fluc_abs_per_ex.append(sum([abs(num) for num in ppl_fluc_not_on_train])/len(ppl_fluc_not_on_train))
            overall_ppl_drop_per_ex.append((1-ppl[-1]/ppl[0])*100)
            forget_ratio_per_ex.append(sum(forget_ratio)/len(forget_ratio))
        
    avg_ppl_drop_on_train = sum(avg_ppl_drop_per_ex)/len(avg_ppl_drop_per_ex)
    avg_ppl_fluc_stdev_not_on_train = sum(avg_ppl_fluc_stdev_per_ex)/len(avg_ppl_fluc_stdev_per_ex)
    avg_ppl_fluc_abs_not_on_train = sum(avg_ppl_fluc_abs_per_ex)/len(avg_ppl_fluc_abs_per_ex)
    avg_forget_ratio = sum(forget_ratio_per_ex)/len(forget_ratio_per_ex)
    
    if remove_outliers:
        overall_ppl_drop_per_ex = remove_outliers_iqr(overall_ppl_drop_per_ex)
        
    avg_overall_ppl_drop = sum(overall_ppl_drop_per_ex)/len(overall_ppl_drop_per_ex)
    
    result = {
        'ppl_drop_on_train': (avg_ppl_drop_on_train, avg_ppl_drop_per_ex),
        'ppl_fluc_stdev_not_on_train': (avg_ppl_fluc_stdev_not_on_train, avg_ppl_fluc_stdev_per_ex),
        'ppl_fluc_abs_not_on_train': (avg_ppl_fluc_abs_not_on_train, avg_ppl_fluc_abs_per_ex),
        'forget_ratio': (avg_forget_ratio, forget_ratio_per_ex),
        'overall_ppl_drop': (avg_overall_ppl_drop, overall_ppl_drop_per_ex) 
        }
    
    
    return result


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

    results=[load_json(os.path.join(args.base_dir, exp_name)) for exp_name in args.exp_name]
    with open(os.path.join(args.base_dir, args.text_log), 'r') as f:
        text_log = json.load(f)


    dataset_fpath='/home/work/parrot/trl-pretrain/custom_knowledge/custom_knowledge_200.json'
    pre = load_json(dataset_fpath)

    ref_texts=[]
    for i, d in enumerate(pre):
        text = d["definition"][:-12]
        ref_texts.append(text)

    train_indices=[[] for i in range(len(ref_texts))]
    for step, texts in tqdm(text_log.items()):
        if int(step)<=100:
            continue
        for text in texts:
            try:
                index = ref_texts.index(text.strip())
                train_indices[index].append(int(step))
            except:
                continue


    if args.mode=='draw_figures':
        plot_indices = range(156,196)
        plot_ppl_with_trained_at(results, save_dir=args.save_dir, train_indices=train_indices)



    
    elif args.mode=='measure_scores':
        # measure_indices = list(range(len(per_ex)))
        result = measure_scores(results[0], train_indices)
        # assert len(avg_ppl_drop_per_ex)==len(measure_indices)
        # print(f"\n\n################################################################################\n \
        #         avg_ppl_drop_per_ex:\n\n{result['ppl_drop_on_train'][1]}\n\n \
        #         avg_ppl_drop: {result['ppl_drop_on_train'][0]} \
        #         \n\navg_ppl_fluc_abs_per_ex:\n\n{result['ppl_fluc_abs_not_on_train'][1]}\n\n \
        #         avg_ppl_fluc_abs: {result['ppl_fluc_abs_not_on_train'][0]}\n\n \
        #         \n\navg_ppl_fluc_stdev_per_ex:\n\n{result['ppl_fluc_stdev_not_on_train'][1]}\n\n \
        #         avg_ppl_fluc_stdev: {result['ppl_fluc_stdev_not_on_train'][0]}\n\n \
        #         \n\navg_forget_ratio_per_ex:\n\n{result['forget_ratio'][1]}\n\n \
        #         avg_forget_ratio: {result['forget_ratio'][0]}\n\n \
        #         overall_ppl_drop: {result['overall_ppl_drop'][0]} \
        #         \n################################################################################\n\n")
        # print(f"\n\n################################################################################\n \
        #         avg_ppl_drop: {result['ppl_drop_on_train'][0]}\n\n \
        #         avg_ppl_fluc_abs: {result['ppl_fluc_abs_not_on_train'][0]}\n\n \
        #         avg_ppl_fluc_stdev: {result['ppl_fluc_stdev_not_on_train'][0]}\n\n \
        #         avg_forget_ratio: {result['forget_ratio'][0]}\n\n \
        #         overall_ppl_drop: {result['overall_ppl_drop'][0]} \
        #         \n################################################################################\n\n")
        
    
    # elif args.mode=='order_examples':
        
    #     def sort_idx(lst):
    #         sorted_pairs = sorted(zip(measure_indices, lst), key=lambda x: x[1], reverse=True)
    #         return [index for index, value in sorted_pairs]
        
    #     result = measure_ppl_drop(per_exs, measure_indices, exclude_pop=True, remove_outliers=False)  
        
    #     sort_by_ppl_drop = sort_idx(result['ppl_drop_on_train'][1])
    #     sort_by_fluc_abs = sort_idx(result['ppl_fluc_abs_not_on_train'][1])
    #     sort_by_fluc_stdev = sort_idx(result['ppl_fluc_stdev_not_on_train'][1])
    #     sort_by_forget_ratio = sort_idx(result['forget_ratio'][1])
    #     sort_by_overall_ppl_drop = sort_idx(result['overall_ppl_drop'][1])
        
    #     print(f"\n\n################################################################################\n \
    #                 avg_ppl_drop_per_ex:\n\n{sort_by_ppl_drop}\n\n \
    #                 \n\navg_ppl_fluc_abs_per_ex:\n\n{sort_by_fluc_abs}\n\n \
    #                 \n\navg_ppl_fluc_stdev_per_ex:\n\n{sort_by_fluc_stdev}\n\n \
    #                 \n\navg_forget_ratio_per_ex:\n\n{sort_by_forget_ratio}\n\n \
    #                 overall_ppl_drop: {sort_by_overall_ppl_drop} \
    #                 \n################################################################################\n\n")
    
    else:
        raise NotImplementedError
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--base_dir', type=str, default="/home/work/parrot/trl-pretrain/results/logs")
    parser.add_argument('--save_dir', type=str, default="test")
    parser.add_argument('--exp_name', nargs='+', required=True)
    parser.add_argument('--text_log', type=str, required=True)
    parser.add_argument('--mode', type=str, default="draw_figures")

    # Parse the arguments
    args = parser.parse_args()

    main(args)