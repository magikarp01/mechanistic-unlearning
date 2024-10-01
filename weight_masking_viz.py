#%%
from turtle import color
import pandas as pd
import json

from matplotlib import colors, pyplot as plt

#%%
model_name = "google/gemma-2-9b"
localization_type = "ap"
SUFFIX = "edit"

localization_types = ["ap", "ct", "random", "manual", "none", "mlp"]

results = {}
for localization_type in localization_types:
    with open(f"results/{model_name.replace('/', '_')}-{localization_type}-results-counterfact-{SUFFIX}.json", "r") as f:
        res = json.load(f)
        results[localization_type] = res

#%%
localization_types = ["ap", "ct", "random", "manual", "none", "mlp"]
adv_res_names = ["Normal", "MC", "Paraphrase", "Neighborhood"]

type_to_name = {
    'ap': 'Localized AP',
    'ct': 'Localized CT',
    'random': 'Random',
    'manual': 'Manual Interp',
    'none': 'Nonlocalized',
    'mlp': "All MLPs"
}

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'k']
markers = ['o', 'v', '^', 's', 'p', '*', 'P']
for adv_res_name in adv_res_names:
    color_idx = 0
    marker_idx = 0
    fig = plt.figure(figsize=(8, 6))
    for localization_type in localization_types:
        forget_accs = []
        maintain_accs = []
        for i in results[localization_type].keys():
            forget_accs.append(results[localization_type][i]['Adversarial'][adv_res_name]["forget"])
            maintain_accs.append(results[localization_type][i]['Adversarial'][adv_res_name]["maintain"])
        plt.scatter(
            forget_accs, 
            maintain_accs, 
            label=f"{type_to_name[localization_type]}", 
            color=colors[color_idx], 
            marker=markers[marker_idx],
            s=50
        )
        color_idx += 1
        marker_idx += 1

    if adv_res_name == "MC":
        adv_res_name = "MCQ"
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.title(f"Editing: {adv_res_name} Forget vs Maintain Accuracy", fontsize=16)
    plt.xlabel(f"{adv_res_name} Forget Accuracy", fontsize=16)
    plt.ylabel(f"{adv_res_name} Maintain Accuracy", fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    fig.savefig(f"results/{adv_res_name}-forget-maintain-acc-{SUFFIX}.pdf", bbox_inches='tight')

#%%
# Normal and MCQ forget vs weights masked

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'k']
markers = ['o', 'v', '^', 's', 'p', '*', 'P']
for adv_res_name in ["Normal", "MC"]:
    color_idx = 0
    marker_idx = 0
    
    fig = plt.figure(figsize=(10, 6))
    for localization_type in localization_types:
        forget_accs = []
        num_weights = []
        for i in results[localization_type].keys():
            forget_accs.append(results[localization_type][i]['Adversarial'][adv_res_name]["forget"])
            num_weights.append(int(i))
        # Make num weights of the form X M
        num_weights = [0] + [f"{int(x/1e6)}M" for x in num_weights[1:]]
        plt.plot(num_weights, forget_accs, label=f"{type_to_name[localization_type]} Forget", color=colors[color_idx], marker=markers[marker_idx])
        color_idx += 1
        marker_idx += 1

    ax = plt.gca()
    ax.set_ylim([0, 1])

    if adv_res_name == "MC":
        adv_res_name = "MCQ"
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title(f"Editing: {adv_res_name} Forget vs Maintain Accuracy", fontsize=16)
    plt.xlabel("Number of Masked Weights", fontsize=16)
    plt.ylabel(f"{adv_res_name} Accuracy", fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.show()
    fig.savefig(f"results/{adv_res_name}-forget-acc-{SUFFIX}.pdf", bbox_inches='tight')
#%%
localization_types = ['ap', 'ct', 'random', 'manual', 'none']
type_to_name = {
    'ap': 'Localized AP',
    'ct': 'Localized CT',
    'random': 'Random',
    'manual': 'Manual Interp',
    'none': 'Nonlocalized'
}
eval_names = ['Adversarial: System Prompt']
forget_sports = ['athlete']#, 'athlete']
forget_sport = 'athlete'

results = {}
for localization_type in localization_types:
    with open(f"results/google_gemma-7b-{localization_type}-results-{forget_sport}.json", "r") as f:
        res = json.load(f)
        results[localization_type] = res
#%%
# Abbreviate numbers using FuncFormatter

def format_func(x):
    x = int(x)
    if x >= 1e6:
        return f"{round(x/1e6, 1)}M"
    elif x >= 1e3:
        return f"{int(x/1e3)}k"
    else:
        return str(int(x))


from matplotlib import pyplot as plt
color_idx = 0
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'v', '^', 's', 'p', '*', 'P']
# all_sports = set(['football', 'baseball', 'basketball'])
all_sports = set(["forget", "maintain"])
for forget_sport in forget_sports:
    fig = plt.figure(figsize=(10, 6))
    for eval_name in eval_names:
        for localization_type in localization_types:
            subeval_name = 'MC'
            subeval_sport = 'forget' #forget_sport
            # for subeval_sport in ['baseball', 'basketball', 'football']:
            subeval_metric = []
            maintain_acc = []
            for i in results[localization_type][forget_sport].keys():
                subeval_metric.append(
                    results[localization_type][forget_sport][i][eval_name][subeval_name][subeval_sport]
                )

            for i in results[localization_type][forget_sport].keys():
                m_acc = 0
                for other_sport in all_sports - set([forget_sport]):
                    m_acc += results[localization_type][forget_sport][i]['Adversarial: System Prompt']['MC'][other_sport]
                maintain_acc.append(m_acc / 2)

            xvals = [int(x) for x in results[localization_type][forget_sport].keys()]
            xlabels = [format_func(x) for x in results[localization_type][forget_sport].keys()]
            plt.plot(
                xvals,
                subeval_metric, 
                label=f'{type_to_name[localization_type]} forget accuracy',
                color=colors[color_idx],
                marker=markers[color_idx]
            )
            # plt.plot(
            #     xvals,
            #     maintain_acc, 
            #     # 'm--',
            #     label=f'{type_to_name[localization_type]} maintain accuracy',
            #     color=colors[color_idx],
            #     marker=markers[color_idx]
            # )
            color_idx += 1
            plt.xticks(ticks=xvals, labels=xlabels, rotation=-50, fontsize=16)
            plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 2_400_000])
    plt.legend(fontsize=12, loc='lower left')
    # plt.ylabel(f'{subeval_sport.title()} Accuracy', fontsize=16)
    plt.ylabel(f'Forget Accuracy', fontsize=16)
    plt.xlabel('Number of Masked Weights', fontsize=16)
    plt.grid()
    plt.title(f'{forget_sport.title()} Accuracy vs Num. Masked Weights', fontsize=16)
    plt.show()
    # Save as PDF
    # plt.savefig(f"results/{forget_sport}-{eval_name}-{subeval_name}.pdf")
    fig.savefig("athlete-forget-acc.pdf", bbox_inches='tight')
            

# %%
all_sports = set(['football', 'baseball', 'basketball'])
markers = ['o', 'v', '^', '*', 'p', 's', 'P']
# Pareto plots
for forget_sport in forget_sports:
    # Need to plot 'Normal' {forget_sport} eval against:
    # - Other sports 'Normal' eval
    # - Tennis side effect eval
    # - Pile and OWT Side Effects Cross Entropy eval

    # Normal eval
    # for localization_type in localization_types:
    #     normal_eval = []

    #     for i in results[localization_type][forget_sport].keys():
    #         normal_eval.append(
    #             results[localization_type][forget_sport][i][eval_name][subeval_name][subeval_sport]
    #             results[localization_type][forget_sport]['Normal'][eval_name][forget_sport]
    #         )

    fig = plt.figure()
    # Tennis Side Effects
    color_idx = 0
    for localization_type in localization_types:
        forget_acc = []
        maintain_acc = []
        for i in results[localization_type][forget_sport].keys():
            forget_acc.append(
                results[localization_type][forget_sport][i]['Adversarial: System Prompt']['MC']['forget'] # forget_sport
            )

        tennis_eval = []
        for i in results[localization_type][forget_sport].keys():
            tennis_eval.append(
                results[localization_type][forget_sport][i]['Side Effects']['General']['MMLU']
            )

        ax = plt.gca()
        ax.set_ylim([0.3, .7])
        ax.set_xlim([0, 1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.scatter(forget_acc, tennis_eval, label=f'{type_to_name[localization_type]}', marker=markers[color_idx])
        plt.xlabel(f"Accuracy: Athlete MCQ", fontsize=16)
        plt.ylabel(f"MMLU Accuracy", fontsize=16)
        plt.title(f"Athlete MCQ vs MMLU Accuracy", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.plot()
        color_idx += 1
    plt.show()
    fig.savefig("athlete-mc-mmlu.pdf", bbox_inches='tight')
        

# %%
