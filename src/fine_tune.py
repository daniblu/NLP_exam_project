'''
Code slightly adapted from https://towardsai.net/p/l/transformers-for-multi-regression-part2
'''

from pathlib import Path
import pandas as pd
import numpy as np
import math
import pickle
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    AutoModelForSequenceClassification, AutoConfig, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer, TrainingArguments
)

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modelcard import parse_log_history

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-bs", type=int, required=True, help="training batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5, help="max learning rate (scientific notation) during schedule, default=2e-5")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="number of epochs, default=15")
    parser.add_argument("--seed", "-s",type=int, default=50, help="random state for model training, default=50")
    args = parser.parse_args()

    return(args)

class RegressionModel(nn.Module):
    '''
    A custom model that takes a pretrained model and adds a dropout layer and a linear layer on top.
    '''

    def __init__(self, model_name, label_cols, device):
        super(RegressionModel, self).__init__()
        self.device = device
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.config.hidden_size, len(label_cols))
        
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, # out should be of type SequenceClassifierOutput
                        attention_mask=attention_mask, 
                        output_hidden_states=True)
        cls_token = out.hidden_states[-1][:, 0, :].to(self.device)
        out = self.drop(cls_token )
        outputs = self.fc(out)
        return SequenceClassifierOutput(logits=outputs)

class RMSELoss(nn.Module):
    """
    Defines the loss function to be fed into the CustomTrainer.
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets))
        return loss

class CustomTrainer(Trainer):
    '''
    A custom trainer class that overwrites the compute_loss method of Trainer to use RMSE loss
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], inputs['attention_mask']) # model outputs are of type SequenceClassifierOutput
        loss_func = RMSELoss()
        loss = loss_func(outputs.logits.float(), inputs['labels'].float()) # predictions, targets
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    '''
    A custom function that allows calculating the RMSE of each of the six metrics separately.
    '''
    predictions, labels = eval_pred
    colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))
    res = {
        f"{analytic.upper()}_RMSE" : colwise_rmse[i]
        for i, analytic in enumerate(['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity'])
    }
    res["MCRMSE"] = np.mean(colwise_rmse)
    return res

def main(batch_size, learning_rate, epochs, seed):

    # configuration
    CONFIG = {
        "model_name": "distilbert-base-uncased",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "max_length": 512,
        "train_batch_size": batch_size,
        "valid_batch_size": 158,
        "epochs": epochs,
        "max_grad_norm": 1000,
        "weight_decay": 1e-6, # Btwn 0-0.1. "The higher the value, the less likely your model will overfit. However, if set too high, your model might not be powerful enough."
        "learning_rate": learning_rate, # the BERT paper used 5e-5, 4e-5, 3e-5, and 2e-5 for fine-tuning
        "loss_type": "rmse",
        "n_accumulate" : 1,
        "label_cols" : ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity'],
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.001,
        "seed": seed   
    }

    # paths
    data_path = Path(__file__).parents[1] / 'story_eval_dataset.pkl'
    models_path = Path(__file__).parents[1] / 'models' / f'bs{batch_size}_lr{learning_rate}_e{epochs}'

    # check if models_path exists, if not create it
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)

    # tokenize data

    def tokenize(examples): # must be defined within main() in order for CONFIG to be recognized and because only examples may be given as input
        '''
        A function to be used with the map method of the dataset class. 
        Tokenizes the text and returns a dictionary of tensors.
        '''
        labels = examples['label']
        tokens = tokenizer(examples['text'], 
                        padding='max_length', 
                        truncation=True, 
                        max_length=CONFIG['max_length'], 
                        return_tensors='pt',
                        return_attention_mask=True)
        res = {
            'input_ids': tokens['input_ids'].to(CONFIG['device']).squeeze(),
            'attention_mask': tokens['attention_mask'].to(CONFIG['device']).squeeze(),
            'labels': torch.tensor(labels)
        }

        return res

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print('[INFO]: Tokenizing data.')
    for split in dataset.keys():
        dataset[split] = dataset[split].map(tokenize, remove_columns=['model'])

    # calculate bactches per epoch
    bacthes_per_epoch = math.ceil(len(dataset['train'])/(CONFIG['train_batch_size'] * CONFIG['n_accumulate']))

    # define the training arguments
    training_args = TrainingArguments(
        output_dir=models_path,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS, # save checkpoint for each save_steps
        eval_steps=bacthes_per_epoch, # compute metrics after each epoch
        save_steps=bacthes_per_epoch,
        logging_steps=bacthes_per_epoch,
        logging_first_step=False,
        logging_dir=models_path, 
        per_device_train_batch_size=CONFIG['train_batch_size'],
        per_device_eval_batch_size=CONFIG['valid_batch_size'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        gradient_accumulation_steps=CONFIG['n_accumulate'],
        use_cpu=True if CONFIG['device'] == 'cpu' else False,
        use_ipex=True if CONFIG['device'] == 'cpu' else False,
        bf16=True if CONFIG['device'] == 'cpu' else False,
        seed=CONFIG['seed'],
        group_by_length=True,
        max_grad_norm=CONFIG['max_grad_norm'],
        metric_for_best_model='eval_MCRMSE',
        load_best_model_at_end=True, # always save best checkpoint at end of training. May exceed save_total_limit if best and last model are different.
        greater_is_better=False,
        save_total_limit=1,
        label_names=["labels"] 
    )

    # data collator for dynamic padding
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # define early stopping criteria
    # note, if early stop occurs, the model will not be saved in the ckeckpoint (https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167)
    # thus trainer.save_model() is necessary
    early_stop = EarlyStoppingCallback(early_stopping_patience = CONFIG['early_stopping_patience'], 
                                       early_stopping_threshold = CONFIG['early_stopping_threshold'])

    # init model
    model = RegressionModel(model_name=CONFIG['model_name'], label_cols=CONFIG['label_cols'], device=CONFIG['device'])
    model.to(CONFIG['device'])

    # count number of trainable params (total and in head)
    #total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #modelhead = nn.Linear(model.config.hidden_size, len(CONFIG['label_cols']))
    #head = sum(p.numel() for p in modelhead.parameters() if p.requires_grad)

    # SET THE OPITMIZER AND THE SCHEDULER
    # no decay for bias and normalization layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
    {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], # get all the params except those in no_decay
            "weight_decay": CONFIG['weight_decay'],
    },
    {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], # get all the params that are in no_decay
            "weight_decay": 0.0,
    },
    ]
    optimizer = AdamW(optimizer_parameters, lr=CONFIG['learning_rate'])
    
    num_training_steps = bacthes_per_epoch * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1*num_training_steps,
        num_training_steps=num_training_steps
    )

    # init trainer
    trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=collate_fn,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics,
            callbacks=[early_stop])
    
    # train
    print("[INFO]: Training model.")
    trainer.train()

    # save model
    print("[INFO]: Saving model.")
    #trainer.save_model(models_path / 'fine_tuned_model')
    torch.save(model.state_dict(), models_path / 'model_state')

    # save loss history
    log_history = parse_log_history(trainer.state.log_history)
    with open(models_path / 'log_history', 'wb') as f:
        pickle.dump(log_history, f)

    # save configurations
    config = [f'{key}: {value}' for key, value in CONFIG.items()]
    with open(models_path / 'config.txt', 'w') as f:
        f.write('\n'.join(config))

    print("[INFO]: Finished.")


if __name__ == '__main__':
    args = input_parse()
    main(batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs, seed=args.seed)
