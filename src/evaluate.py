# TODO: Look at absolute error dist within each metric. 
#       If symmetric dists, summarise with mean - either in dist plots or in table.     
#       Identify the best hyperparameters using MAE or some other method... summed absolute error? 
#           - How well are the individual scores predicted?
#       Stratify test stories by absolute error.
#           - Are the outputs of certain models overrepresented in higher/lower strata?
#           - That is, are some model-stories harder to evaluate by the regression model?
#       Identify stories where the reg model was very wrong. 

print('[INFO]: Importing libraries and loading materials')
import math
import pandas as pd

# loading data
data_path = Path(__file__).parents[1] / 'story_eval_dataset_dict.pkl'

with open(data_path, 'rb') as f:
    dataset = pickle.load(f)



print('[INFO]: Making predictions')
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

n_batches = math.ceil(len(raw_test_ds)/BATCH_SIZE)
y_preds = []

for i in range(nb_batches):
    input_texts = raw_test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["text"]
    input_labels = raw_test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["score"]
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
    y_preds += model(**encoded).logits.reshape(-1).tolist()

df = pd.DataFrame([raw_test_ds["text"], raw_test_ds["score"], y_preds], ["Text", "Score", "Prediction"]).T
