'''
This script loads the HANNA dataset and prepares it for the model. 
This involves averaging human annotations, train-test splitting, and organsing the data in a Dataset object.
Some plots are also created to inspect the splits.
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
    #Relevance=('Relevance', 'mean'),
    Coherence=('Coherence', 'mean'),
    Empathy=('Empathy', 'mean'),
    Surprise=('Surprise', 'mean'),
    Engagement=('Engagement', 'mean'),
    Complexity=('Complexity', 'mean')
    )

    # combine scores to single label
    y = data_agg.iloc[:, 5:11].agg(func=np.array, axis=1)

    # isolate stories
    X = data_agg.iloc[:, [3,4]]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    # val test split
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=55)

    # count models in val set and test set and merge series
    X_val_counts = X_val['Model'].value_counts()
    X_test_counts = X_test['Model'].value_counts()
    df = pd.concat([X_val_counts, X_test_counts], axis=1, keys=['val', 'test'])
    print('Count of each model in val and test set:\n', df)

    # plot distribution of scores for each model in test set
    metrics = ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']

    coherence_score = [score[0] for score in y_test]
    empathy_score = [score[1] for score in y_test]
    surprise_score = [score[2] for score in y_test]
    engagement_score = [score[3] for score in y_test]
    complexity_score = [score[4] for score in y_test]
    score_list = [coherence_score, empathy_score, surprise_score, engagement_score, complexity_score]

    df = pd.DataFrame({'Model': X_test['Model'].tolist() * 5, 'Scores': np.concatenate(score_list)})

    ags_models = ['Human', 'BertGeneration', 'CTRL', 'RoBERTa', 'XLNet', 'GPT', 'GPT-2', 'GPT-2 (tag)', 'HINT', 'Fusion', 'TD-VAE']

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=df, x='Model', y='Scores', 
                fill=False, whis=(0,100), color='black', linewidth=1,
                order=ags_models)
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    ax.yaxis.grid(True)
    fig.savefig(plot_path / 'scores_of_models_in_test.png')

    # plot distribution of scores in train and validation sets for each metric
    fig, ax = plt.subplots(5, 3, figsize=(13, 18))
    for i, metric in enumerate(metrics):
        sns.histplot([score[i] for score in y_train], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 0]).set(xlabel=None)
        sns.histplot([score[i] for score in y_val], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 1]).set(xlabel=None)
        sns.histplot([score[i] for score in y_test], bins=np.arange(0, 5+(1/3), 1/3), ax=ax[i, 2]).set(xlabel=None)
        ax[i, 0].set_xlim(1, 5)
        ax[i, 0].set_ylabel(metric, rotation=0, fontsize=20)
        ax[i, 0].yaxis.set_label_coords(-0.37, 0.5)
        ax[i, 1].set_xlim(1, 5)
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_xlim(1, 5)
        ax[i, 2].set_ylabel('')

    for ax, col in zip(ax[0], ['Train', 'Validation', 'Test']):
        ax.set_title(col, fontsize=20)

    fig.tight_layout()
    plt.savefig(plot_path / 'splits_score_dist.png')

    # create Dataset objects
    train_dict = {'label': y_train, 'text': X_train['Story'], 'model': X_train['Model']}
    ds_train = Dataset.from_dict(train_dict)
    test_dict = {'label': y_test, 'text': X_test['Story'], 'model': X_test['Model']}
    ds_test = Dataset.from_dict(test_dict)
    val_dict = {'label': y_val, 'text': X_val['Story'], 'model': X_val['Model']}
    ds_val = Dataset.from_dict(val_dict)

    # combine into dictionary
    dataset = {'train': ds_train, 'test': ds_test, 'validation': ds_val}

    # save dataset as pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f'[INFO]: Finished. Dataset saved in {output_path.name}')