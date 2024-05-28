# DSPRO2 Outfit Recommender

## Requirements
- Pipenv https://pypi.org/project/pipenv

## Steps after cloning the repository

1. Download the dataset from ```<path-to-be-defined>``` and unzip it in the folder <project-root>/datasets
2. Execute ```pipenv install``` in the project root

3. Generate requirements.txt File ``` pipenv requirements > requirements.txt ```

## How to train the model on GPU hub
```
cd <project-root> 
```
Upload the newest requirements.txt File to the GPUhub. 
```
pip install -r requirements.txt 
```


```
export PYTHONPATH="$PYTHONPATH:$PWD"
```


```
cd src && python ./train_dino_v2_based_model.py
```


