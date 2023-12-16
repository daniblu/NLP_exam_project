from pathlib import Path
import pickle

models_folder = Path(Path(__file__).parents[1] / 'models')
output_folder = Path(Path(__file__).parents[1])

loss_dict = {}

for folder in models_folder.iterdir():
    if folder.is_dir():
        log_history_path = folder / 'log_history'
        if log_history_path.exists():
            with open(log_history_path, 'rb') as f:
                log_history = pickle.load(f)
            val_loss = [log['Validation Loss'] for log in log_history[1]]
            val_loss = val_loss[-1] if len(val_loss) > 0 else val_loss
            train_loss = [log['Training Loss'] for log in log_history[1]]
            train_loss = train_loss[-1] if len(train_loss) > 0 else train_loss
            loss_dict[folder.name] = [val_loss, train_loss]

# sort val_loss_dict by first value in ascending order
loss_dict = dict(sorted(loss_dict.items(), key=lambda item: item[1][0]))
#loss_dict = dict(sorted(loss_dict.items(), key=lambda item: item[1]))

# save val_loss_dict as txt file
with open(output_folder / 'loss_table.txt', 'w') as f:
    for key, val in loss_dict.items():
        f.write(f'{key}| val_loss:{val[0]} | train_loss:{val[1]}\n')

