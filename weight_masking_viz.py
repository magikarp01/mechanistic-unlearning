#%%
import pandas as pd
import json
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
    plt.ylabel(f'{eval_name} {subeval_name}: {subeval_sport}')
    plt.xlabel('Number of Masked Weights')
    plt.grid()
    plt.title(f'{forget_sport} Accuracy vs Num. Masked Weights')
    plt.show()
            

# %%
