#%%
import pandas as pd
import json

from matplotlib import pyplot as plt
#%%
localization_types = ['ap', 'ct', 'random', 'manual', 'none']
eval_names = ['Adversarial: System Prompt']
forget_sports = ['baseball', 'basketball', 'football']

results = {}
for localization_type in localization_types:
    with open(f"results/google_gemma-7b-{localization_type}-results.json", "r") as f:
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
for forget_sport in forget_sports:
    for eval_name in eval_names:
        for localization_type in localization_types:
            subeval_name = 'Normal'
            subeval_sport = forget_sport
            # for subeval_sport in ['baseball', 'basketball', 'football']:
            subeval_metric = []
            for i in results[localization_type][forget_sport].keys():
                subeval_metric.append(
                    results[localization_type][forget_sport][i][eval_name][subeval_name][subeval_sport]
                )
            plt.plot(
                [format_func(x) for x in results[localization_type][forget_sport].keys()],
                subeval_metric, 
                label=f'{localization_type}-{subeval_name}-{subeval_sport}'
            )
    plt.legend()
    plt.ylabel(f'{subeval_name}: {subeval_sport.title()}')
    plt.xlabel('Number of Masked Weights')
    plt.grid()
    plt.title(f'{forget_sport.title()} Accuracy vs Num. Masked Weights')
    plt.show()
    # Save as PDF
    plt.savefig(f"results/{forget_sport}-{eval_name}-{subeval_name}.pdf")
            

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
