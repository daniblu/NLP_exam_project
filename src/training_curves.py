from pathlib import Path 
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str, required=True, help="model directory, e.g. bs32_lr1e-05_e15")
    args = parser.parse_args()

    return(args)

def extract_bs_from_model_dir(input_string):
    match = re.search(r'bs(\d+)_lr(\d)e-05_e(\d+)', input_string)
    if match:
        return match.group(1)
    else:
        return None

def extract_lr_from_model_dir(input_string):
    match = re.search(r'bs(\d+)_lr(\de-05)_e(\d+)', input_string)
    if match:
        return match.group(2)
    else:
        return None

def main(model_dir):
    
    # paths
    log_path = Path(__file__).parents[1] / 'models' / model_dir / 'log_history'
    plot_path = Path(__file__).parents[1] / 'plots'

    # load log history
    with open(log_path, 'rb') as f:
        log_history = pickle.load(f)
    
    # extract Training loss and Validation loss from log_history
    train_loss = [log['Training Loss'] for log in log_history[1]]
    val_loss = [log['Validation Loss'] for log in log_history[1]]

    # extract batch size and learning rate from model_dir
    bs = extract_bs_from_model_dir(model_dir)
    lr = extract_lr_from_model_dir(model_dir)

    # number of epochs
    n_epochs = len(train_loss)

    # plot the training and validation loss
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, n_epochs+1), train_loss, label='Training loss')
    ax.plot(np.arange(1, n_epochs+1), val_loss, label='Validation loss')
    title_text = f'Batch size = {bs}, Learning rate = {lr}'
    ax.set_title(title_text, loc='right', y=1.0, pad=-14)
    ax.legend(loc=(0.68, 0.78))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE loss')
    plt.savefig(plot_path / f'loss_{model_dir}.png')

    print(f'[INFO]: Finished. Plotted loss of {model_dir}')

if __name__ == '__main__':
    args = input_parse()
    main(args.model_dir)
