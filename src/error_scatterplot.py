import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str, required=True, help="model directory, e.g. bs32_lr1e-05_e15")
    args = parser.parse_args()

    return(args)

def str_to_array(s):
    '''
    Convert string representation of list to numpy array.
    '''
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace(' ', '')
    s = s.split(',')
    return np.array(s).astype(float)

def main(model_dir):
    
    data_path = Path(__file__).parents[1] / 'test predictions' / f'test_predictions_{model_dir}.csv'
    plot_path = Path(__file__).parents[1] / 'plots'

    # load data
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

    # split Labels into 5 columns
    df['Coherence_label'] = df['Labels'].apply(lambda x: str_to_array(x)[0])
    df['Empathy_label'] = df['Labels'].apply(lambda x: str_to_array(x)[1])
    df['Surprise_label'] = df['Labels'].apply(lambda x: str_to_array(x)[2])
    df['Engagement_label'] = df['Labels'].apply(lambda x: str_to_array(x)[3])
    df['Complexity_label'] = df['Labels'].apply(lambda x: str_to_array(x)[4])

    # split Error into 5 columns
    df['Coherence_error'] = df['Error'].apply(lambda x: x[0])
    df['Empathy_error'] = df['Error'].apply(lambda x: x[1])
    df['Surprise_error'] = df['Error'].apply(lambda x: x[2])
    df['Engagement_error'] = df['Error'].apply(lambda x: x[3])
    df['Complexity_error'] = df['Error'].apply(lambda x: x[4])

    # split Absolute Error into 5 columns
    df['Coherence_abs_error'] = df['Absolute Error'].apply(lambda x: x[0])
    df['Empathy_abs_error'] = df['Absolute Error'].apply(lambda x: x[1])
    df['Surprise_abs_error'] = df['Absolute Error'].apply(lambda x: x[2])
    df['Engagement_abs_error'] = df['Absolute Error'].apply(lambda x: x[3])
    df['Complexity_abs_error'] = df['Absolute Error'].apply(lambda x: x[4])

    # plot error versus label for each metric
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    sns.scatterplot(x='Coherence_label', y='Coherence_error', data=df, ax=axes[0])
    sns.scatterplot(x='Empathy_label', y='Empathy_error', data=df, ax=axes[1])
    sns.scatterplot(x='Surprise_label', y='Surprise_error', data=df, ax=axes[2])
    sns.scatterplot(x='Engagement_label', y='Engagement_error', data=df, ax=axes[3])
    sns.scatterplot(x='Complexity_label', y='Complexity_error', data=df, ax=axes[4])

    axes[0].set_ylabel('Prediction error', fontsize=18)
    axes[0].yaxis.set_label_coords(-0.3, 0.5)
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    axes[3].set_ylabel('')
    axes[4].set_ylabel('')

    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel('True label', fontsize=18)
    axes[2].xaxis.set_label_coords(0.4, -0.22)
    axes[3].set_xlabel('')
    axes[4].set_xlabel('')

    axes[0].set_title('Coherence')
    axes[1].set_title('Empathy')
    axes[2].set_title('Surprise')
    axes[3].set_title('Engagement')
    axes[4].set_title('Complexity')

    plt.savefig(plot_path / f'error_vs_label_{model_dir}.png', bbox_inches='tight')

if __name__ == '__main__':
    args = input_parse()
    main(args.model_dir)
    print('[INFO]: Finished. Error scatterplot saved.]')