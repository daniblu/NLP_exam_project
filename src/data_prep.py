'''
This script loads the HANNA dataset and prepares it for the model. 
This involves averaging human annotations, train-test splitting, and evaluating label distributions within splits.
'''

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

if __name__ == '__main__':

    # paths
    data_path = Path(__file__).parents[1] / 'hanna_stories_annotations.csv'
    plot_path = Path(__file__).parents[1] / 'plots'
    output_path = Path(__file__).parents[1] / 'story_eval_dataset.pkl'

    # load data
    data = pd.read_csv(data_path)

    # group data by first 5 columns and average metric columns
    data_agg = data.groupby(data.columns[:5].tolist(), as_index=False).agg(
    #Relevance=('Relevance', 'mean'),
    Coherence=('Coherence', 'mean'),
    Empathy=('Empathy', 'mean'),
    Surprise=('Surprise', 'mean'),
    Engagement=('Engagement', 'mean'),
    Complexity=('Complexity', 'mean')
    )

    # isolate scores
    y = data_agg.iloc[:, 5:10]

    # isolate stories
    X = data_agg.iloc[:, 3]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    # val test split
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=55)

    # plot distribution of scores in train and validation sets for each metric
    metrics = ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']

    # plot distribution of scores in train and validation sets for each metric
    fig, ax = plt.subplots(5, 2, figsize=(8, 24))
    for i, metric in enumerate(metrics):
        sns.histplot(y_train[metric], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 0]).set(xlabel=None)
        sns.histplot(y_val[metric], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 1]).set(xlabel=None)
        ax[i, 0].set_title(f'Train_{metric}')
        ax[i, 0].set_xlim(1, 5)
        ax[i, 1].set_title(f'Validation_{metric}')
        ax[i, 1].set_xlim(1, 5)
    plt.savefig(plot_path / 'train_val_score_dist.png')

    # plot distribution of scores in train and test sets for each metric
    fig, ax = plt.subplots(5, 2, figsize=(8, 24))
    for i, metric in enumerate(metrics):
        sns.histplot(y_train[metric], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 0]).set(xlabel=None)
        sns.histplot(y_test[metric], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 1]).set(xlabel=None)
        ax[i, 0].set_title(f'Train_{metric}')
        ax[i, 0].set_xlim(1, 5)
        ax[i, 1].set_title(f'Test_{metric}')
        ax[i, 1].set_xlim(1, 5)
    plt.savefig(plot_path / 'train_test_score_dist.png')

    # combine X and y in dataframe
    train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    validation = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)

    # combine into list
    dataset = [train, validation, test]

    # save dataset as pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)