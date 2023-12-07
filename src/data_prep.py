'''
This script loads the HANNA dataset and prepares it for the model. 
This involves averaging human annotations, train-test splitting, and organsing the data in a Dataset object. 
'''

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
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
    Relevance=('Relevance', 'mean'),
    Coherence=('Coherence', 'mean'),
    Empathy=('Empathy', 'mean'),
    Surprise=('Surprise', 'mean'),
    Engagement=('Engagement', 'mean'),
    Complexity=('Complexity', 'mean')
    )

    # combine scores to single label
    y = data_agg.iloc[:, 5:11].agg(func=np.array, axis=1)

    # isolate stories
    X = data_agg.iloc[:, 3]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)

    # val test split
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=55)

    # plot distribution of scores in train and validation sets for each metric
    metrics = ['Relevance', 'Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']

    fig, ax = plt.subplots(6, 2, figsize=(8, 24))
    for i, metric in enumerate(metrics):
        sns.histplot([score[i] for score in y_train], bins=12, ax=ax[i, 0])
        sns.histplot([score[i] for score in y_val], bins=12, ax=ax[i, 1])
        ax[i, 0].set_title(f'Train_{metric}')
        ax[i, 0].set_xlim(1, 5)
        ax[i, 1].set_title(f'Validation_{metric}')
        ax[i, 1].set_xlim(1, 5)
    plt.savefig(plot_path / 'train_val_score_dist.png')

    # plot distribution of scores in train and test sets for each metric
    fig, ax = plt.subplots(6, 2, figsize=(8, 24))
    for i, metric in enumerate(metrics):
        sns.histplot([score[i] for score in y_train], bins=12, ax=ax[i, 0])
        sns.histplot([score[i] for score in y_test], bins=12, ax=ax[i, 1])
        ax[i, 0].set_title(f'Train_{metric}')
        ax[i, 0].set_xlim(1, 5)
        ax[i, 1].set_title(f'Test_{metric}')
        ax[i, 1].set_xlim(1, 5)
    plt.savefig(plot_path / 'train_test_score_dist.png')

    # create Dataset objects
    train_dict = {'label': y_train, 'story': X_train}
    ds_train = Dataset.from_dict(train_dict)
    test_dict = {'label': y_test, 'story': X_test}
    ds_test = Dataset.from_dict(test_dict)
    val_dict = {'label': y_val, 'story': X_val}
    ds_val = Dataset.from_dict(val_dict)

    # combine into dictionary
    dataset = {'train': ds_train, 'test': ds_test, 'validation': ds_val}

    # save dataset as pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)