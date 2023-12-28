# NLP_exam_project
This repository contains the code for an exam project in the course _Natural Language Processing_ at Aarhus University (see [course description](https://kursuskatalog.au.dk/en/course/119713/Natural-Language-Processing)). The project involves using DistilBERT for story evaluation, by fine-tuning on the HANNA dataset, provided by [Chhun et al. (2022)](https://github.com/dig-team/hanna-benchmark-asg), consisting of pairs of a story and corresponding human ratings on five criteria. Note, the following tutorial is specific to POSIX systems.

## Setup
Enter the following command in the terminal to create an environment with the versioned packages listed in __requirements.txt__:
```bash
bash setup_env.sh
```

Before running any scripts, the environment should be activated:
```bash
source env/bin/activate
```

Should the user wish to experiment within the environment using a Python notebook, the following must be entered to install an environment kernel for the notebook:
```bash
bash setup_kernel.sh
```

## Usage
__Main scripts__
To prepare the HANNA dataset for fine-tuning, run
```
python3 data_prep.py
```
This assumes your working directory is ```NLP_exam_project/src```, although this is not requiered for the script to work as intended. The script produces __story_eval_dataset.pkl__, __splits_score_dist.png__, and __scores_per_model_in_all_splits.png__. Next step is fine-tuning. Some hyperparameters can be set from the terminal including the training batch size, learning rate, and number of epochs. An example is
```
python3 fine_tune.py -bs 4 -lr 3e-5 -e 10
```
For information on flags and defaults, see
```
python3 fine_tune.py -h 
```
The script produces a model folder in ```models```, in this case named __bs4_lr3e-05_e10__, containing:
- a checkpoint folder saved by the ```train```-method from the ```transformers``` library.
- __config.txt__: An overview of hyperparameters of the model along with other information.
- __log_history__: An object containing epoch-wise training history of the model.
- __model_state__: The fine-tuned weights to be loaded for inference.

To load a model of choice, e.g., __bs4_lr3e-05_e10__, and use for prediction on the test set, run
```
python3 test_prediction.py -d bs4_lr3e-05_e10
```
This script produces a _csv_ file of test stories, true scores and predicted scores, saved in __test_predictions__. To create plots for evaluating test performance, run
```
python3 evaluate.py -d bs4_lr3e-05_e10
```
The script produces __abs_error_per_metric_{MODEL NAME}.png__ and __abs_error_per_model_{MODEL NAME}.png__.


__Other scripts__
- ``training_curves``: plot training and validation loss for a model of choice. Produces __loss_{MODEL NAME}.png__.
- ``error_scatterplot.py``: plot a scatterplot of stories with prediction error versus true score for each criterium. Produces __error\_vs\_label\_{MODEL NAME}.png__.
- ``loss_list.py``: Create list of training and validation loss of all models in the __models__ folder. Models are listed in ascening order with respect to validation loss. Produces __loss_table.txt__. 


## Repository overview

```
.
├── models/                         <-- folder for storing fine-tuned models
├── plots/                          <-- folder for storing all plots
├── src/
│   ├── data_prep.py
│   ├── error_scatterplot.py
│   ├── evaluate.py
│   ├── fine_tune.py
│   ├── loss_list.py
│   ├── test_prediction.py
│   └── training_curves.py
├── test_predictions/               <-- folder for storing model predictions on test set
├── .gitignore
├── LICENSE
├── README.md
├── activate.sh
├── hanna_stories_annotations.csv   <-- The HANNA dataset
├── requirements.txt
├── setup_env.sh
└── story_eval_dataset.pkl          <-- Splitted HANNA, prepared for fine_tune.py
```