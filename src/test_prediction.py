print('[INFO]: Importing libraries and loading materials')
from pathlib import Path
import argparse
import pickle
import math
import pandas as pd
import numpy as np

from transformers import AutoTokenizer

import torch

from fine_tune import RegressionModel

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str, required=True, help="model directory, e.g. bs32_lr1e-05_e15")
    args = parser.parse_args()

    return(args)

def main(model_dir):

    # paths
    data_path = Path(__file__).parents[1] / 'story_eval_dataset.pkl'
    model_path = Path(__file__).parents[1] / 'models' / model_dir / 'model_state'
    output_path = Path(__file__).parents[1] / 'test predictions'

    # load data
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    raw_test_ds = dataset['test']
    del dataset

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # configurations - some are needed to init model
    CONFIG = {
        "model_name": "distilbert-base-uncased",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "max_length": 512,
        "label_cols" : ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity'],
        "batch_size": 32 
    }

    # init model
    model = RegressionModel(model_name=CONFIG['model_name'], label_cols=CONFIG['label_cols'], device=CONFIG['device']) #.to('cpu')?

    # load fine-tuned weights into model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(CONFIG['device'])))

    print('[INFO]: Making predictions')
    n_batches = math.ceil(len(raw_test_ds)/CONFIG['batch_size'])
    y_preds = []

    # tokenize and predict in batches
    for i in range(n_batches):
        print(f'[INFO]: Batch {i+1}/{n_batches}')
        input_texts = raw_test_ds[i * CONFIG['batch_size']: (i+1) * CONFIG['batch_size']]["text"]
        input_labels = raw_test_ds[i * CONFIG['batch_size']: (i+1) * CONFIG['batch_size']]["label"]
        encoded = tokenizer(input_texts, 
                            truncation=True, 
                            padding="max_length", max_length=CONFIG["max_length"], 
                            return_tensors="pt", 
                            return_attention_mask=True)
        input_ids = encoded["input_ids"].to('cpu').squeeze()
        attention_masks = encoded["attention_mask"].to('cpu').squeeze()
        y_preds += model(input_ids=input_ids, attention_mask=attention_masks).logits.tolist()

    # save predictions
    df = pd.DataFrame({
        'Model': raw_test_ds['model'],
        'Story': raw_test_ds['text'],
        'Labels': raw_test_ds['label'],
        'Predictions': y_preds
    })

    # save predictions
    df.to_csv(output_path / f'test_predictions_{model_dir}.csv', index=False)

    print('[INFO]: Finished. Test predictions saved.')

if __name__ == '__main__':
    args = input_parse()
    main(args.model_dir)