#%%
import pandas as pd
import json

from matplotlib import colors, pyplot as plt
#%%
localization_types = ['ap', 'ct', 'random', 'manual', 'none']
eval_names = ['Adversarial: System Prompt']
forget_sports = ['basketball']#, 'athlete']
forget_sport = 'basketball'

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
all_sports = set(['football', 'baseball', 'basketball'])
# all_sports = set(["forget", "maintain"])
for forget_sport in forget_sports:
    fig = plt.figure(figsize=(10, 6))
    for eval_name in eval_names:
        for localization_type in localization_types:
            subeval_name = 'MC'
            subeval_sport = forget_sport
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
                label=f'{localization_type} forget accuracy',
                color=colors[color_idx]
            )
            # plt.plot(
            #     xvals,
            #     maintain_acc, 
            #     # 'm--',
            #     label=f'{localization_type} maintain accuracy',
            #     color=colors[color_idx]
            # )
            color_idx += 1
            plt.xticks(ticks=xvals, labels=xlabels, rotation=-45, fontsize=16)
            plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.legend(fontsize=12, loc='lower right')
    # plt.ylabel(f'{subeval_sport.title()} Accuracy', fontsize=16)
    plt.ylabel(f'Maintain Accuracy', fontsize=16)
    plt.xlabel('Number of Masked Weights', fontsize=16)
    plt.grid()
    plt.title(f'{forget_sport.title()} Accuracy vs Num. Masked Weights', fontsize=16)
    plt.show()
    # Save as PDF
    # plt.savefig(f"results/{forget_sport}-{eval_name}-{subeval_name}.pdf")
    # fig.savefig("basketball-maintain-acc.pdf", bbox_inches='tight')
            

# %%
all_sports = set(['football', 'baseball', 'basketball'])
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


    # Tennis Side Effects
    for localization_type in localization_types:
        forget_acc = []
        maintain_acc = []
        for i in results[localization_type][forget_sport].keys():
            forget_acc.append(
                results[localization_type][forget_sport][i]['Adversarial: System Prompt']['Normal'][forget_sport]
            )

        tennis_eval = []
        for i in results[localization_type][forget_sport].keys():
            tennis_eval.append(
                results[localization_type][forget_sport][i]['Side Effects']['Sports Answers']['tennis']
            )
        plt.scatter(forget_acc, tennis_eval, label=f'{localization_type}-Tennis')
        plt.xlabel(f"Accuracy: MC Eval")
        plt.ylabel(f"Side Effect: Tennis Accuracy")
        plt.title(f"{forget_sport} MC vs Tennis Side Effect")
        plt.legend()
        plt.grid()
        plt.plot()
    plt.show()
        

# %%
