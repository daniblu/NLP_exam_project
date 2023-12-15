from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str, required=True, help="model directory, e.g. bs32_lr1e-05_e15")
    args = parser.parse_args()

    return(args)

def str_to_array(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace(' ', '')
    s = s.split(',')
    return np.array(s).astype(float)

def main(model_dir):

    # paths
    data_path = Path(__file__).parents[1] / f'test_predictions_{model_dir}.csv'
    plot_path = Path(__file__).parents[1] / 'plots'

    # load test_predictions.csv
    df = pd.read_csv(data_path)

    # calculate absolute error
    Errors = []
    AErrors = []

    for row in range(len(df)):
        
        predictions = df.loc[row, 'Predictions']
        labels = df.loc[row, 'Labels']

        predictions = str_to_array(predictions)
        labels = str_to_array(labels)

        Error = predictions - labels
        Errors.append(Error)
        AError = abs(predictions - labels)
        AErrors.append(AError)

    df['Error'] = Errors
    df['Absolute Error'] = AErrors

    # plot boxplot of each metric's absolute error
    metrics = ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']

    coherence_error = [error[0] for error in AErrors]
    empathy_error = [error[1] for error in AErrors]
    surprise_error = [error[2] for error in AErrors]
    engagement_error = [error[3] for error in AErrors]
    complexity_error = [error[4] for error in AErrors]
    df['Coherence Error'] = coherence_error
    df['Empathy Error'] = empathy_error
    df['Surprise Error'] = surprise_error
    df['Engagement Error'] = engagement_error
    df['Complexity Error'] = complexity_error
    error_list = [coherence_error, empathy_error, surprise_error, engagement_error, complexity_error]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(error_list, medianprops=dict(color='black'))
    ax.set_xticklabels(metrics)
    ax.yaxis.grid(True)
    ax.set_ylabel('Absolute error')
    fig.savefig(plot_path / f'abs_error_per_metric_{model_dir}.png', bbox_inches='tight')

    # find max absolute error
    lists = df['Absolute Error'].apply(lambda x: x.tolist())
    all_abs_errors = [item for sublist in lists for item in sublist]
    max_abs_error = max(all_abs_errors)

    # plot boxplot of each metric's absolute error for each model
    ags_models = ['Human', 'BertGeneration', 'CTRL', 'RoBERTa', 'XLNet', 'GPT', 'GPT-2', 'GPT-2 (tag)', 'HINT', 'Fusion', 'TD-VAE']
    metric_errors = df.iloc[:, 6:].columns

    fig, axs = plt.subplots(len(metric_errors), 1, figsize=(12, 20))

    for j, metric_error in enumerate(metric_errors):
        sns.stripplot(data=df, x='Model', y=metric_error, 
                    jitter=False, s=4, color='#1f77b4',
                    order=ags_models,
                    ax=axs[j])
        sns.boxplot(data=df, x='Model', y=metric_error, 
                    fill=False, whis=(0,100), color='black', linewidth=1,
                    order=ags_models,
                    ax=axs[j])
        axs[j].set_xlabel('')
        axs[j].set_ylabel('Absolute error')
        axs[j].set_ylim(0, max_abs_error+0.1)
        axs[j].yaxis.grid(True)
        axs[j].set_title(f'{metric_error.split()[0]}', y=1.0, pad=-14)
    fig.savefig(plot_path / f'abs_error_per_model_{model_dir}.png', bbox_inches='tight')

    print(f'[INFO]: Finished. Evaluation plots saved for {model_dir}')

if __name__ == '__main__':
    args = input_parse()
    main(args.model_dir)