'''
Code slightly adapted from https://towardsai.net/p/l/transformers-for-multi-regression-part2
'''

print("[INFO]: Importing libraries.")
from pathlib import Path
import numpy as np
import pickle

import torch
import torch.nn as nn

from transformers import (
    AutoModelForSequenceClassification, AutoConfig, 
    AutoTokenizer,
    AdamW, get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Trainer, TrainingArguments
)

from transformers.modeling_outputs import SequenceClassifierOutput

print("[INFO]: Defining custom classes and functions.")

# configuration
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    #"dropout": random.uniform(0.01, 0.60),
    "max_length": 512,
    "train_batch_size": 8,
    "valid_batch_size": 8, # 16 originally
    "epochs": 3,
    #"folds" : 3,
    "max_grad_norm": 1000,
    "weight_decay": 1e-6, # Btwn 0-0.1. "The higher the value, the less likely your model will overfit. However, if set too high, your model might not be powerful enough."
    "learning_rate": 2e-5,
    "loss_type": "rmse",
    "n_accumulate" : 1,
    "label_cols" : ['Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity'], 
    
}

def tokenize(examples):
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
        'input_ids': tokens['input_ids'].to(CONFIG.get('device')).squeeze(),
        'attention_mask': tokens['attention_mask'].to(CONFIG.get('device')).squeeze(),
        'labels': torch.tensor(labels)
    }

    return res

def compute_metrics(eval_pred):
    '''
    A custom function that allows calculating the RMSE of each of the six metrics separately.
    '''
    predictions, labels = eval_pred
    colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))
    res = {
        f"{analytic.upper()}_RMSE" : colwise_rmse[i]
        for i, analytic in enumerate(CONFIG["label_cols"])
    }
    res["MCRMSE"] = np.mean(colwise_rmse)
    return res

# define the model
class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.config.hidden_size, len(CONFIG['label_cols']))
        
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, # out should be of type SequenceClassifierOutput
                        attention_mask=attention_mask, 
                        output_hidden_states=False)
        out = self.drop(out)
        outputs = self.fc(out) # outputs should be regression scores
        return SequenceClassifierOutput(logits=outputs)


# defining the loss function to feed into the trainer
class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        return loss


# make a custom trainer class that overwrites the compute_loss of Trainer to use RMSE loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], inputs['attention_mask']) # model outputs are of type SequenceClassifierOutput
        loss_func = RMSELoss()
        loss = loss_func(outputs.logits.float(), inputs['labels'].float()) # predictions, targets... is .float() necessary?
        return (loss, outputs) if return_outputs else loss



if __name__ == '__main__':

    # set seed
    SEED = 50

    # paths
    data_path = Path(__file__).parents[1] / 'story_eval_dataset_dict.pkl'
    models_path = Path(__file__).parents[1] / 'models'

    # define the training arguments
    training_args = TrainingArguments(
        output_dir=models_path,
        evaluation_strategy="epoch",
        per_device_train_batch_size=CONFIG['train_batch_size'],
        per_device_eval_batch_size=CONFIG['valid_batch_size'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        gradient_accumulation_steps=CONFIG['n_accumulate'],
        use_cpu=True if CONFIG['device'] == 'cpu' else False, # not sure about this
        use_ipex=True if CONFIG['device'] == 'cpu' else False, # not sure about this
        bf16=True if CONFIG['device'] == 'cpu' else False, # not sure about this
        seed=SEED,
        group_by_length=True,
        max_grad_norm=CONFIG['max_grad_norm'],
        metric_for_best_model='eval_MCRMSE',
        load_best_model_at_end=True,
        greater_is_better=False,
        save_strategy="epoch",
        save_total_limit=1,
        #report_to="wandb",
        label_names=["labels"] 
    )

    # tokenize data
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    for split in dataset.keys():
        dataset[split] = dataset[split].map(tokenize)

    # data collator for dynamic padding
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # init model
    model = FeedBackModel(CONFIG['model_name'])
    model.to(CONFIG['device'])

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
    
    num_training_steps = (len(dataset['train']) * CONFIG['epochs']) // (CONFIG['train_batch_size'] * CONFIG['n_accumulate'])
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
            compute_metrics=compute_metrics)
    
    # train
    print("[INFO]: Training model.")
    trainer.train()

    print("[INFO]: Finished.")

    
    # TODO: what's a warmup scheduler and why is recommended for fine-tuning?
    # TODO: Padding both in custom iterator AND custom trainer?
    # TODO: plots?
    # TODO: How to evaluate?