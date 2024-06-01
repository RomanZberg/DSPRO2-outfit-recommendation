# DSPRO2 Outfit Recommender

## Requirements

- Pipenv https://pypi.org/project/pipenv

## Steps after cloning the repository

1. Download the dataset from ```<path-to-be-defined>``` and unzip it in the folder <project-root>/datasets
2. Execute ```pipenv install``` in the project root

## How to train the model on GPU hub

### On your computer

```
cd <project-root> 
```

```
pipenv requirements > requirements.txt
```

Then upload the requirements.txt file to gpuhub.

### On GPU hub

Open terminal and execute the following commands.

```
pip install -r requirements.txt 
```

```
cd <project-root> 
```

```
export PYTHONPATH="$PYTHONPATH:$PWD"
```

```
cd src && python ./train_dino_v2_based_model.py
```


