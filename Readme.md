# DSPRO2 Outfit Recommender

## Requirements

- Pipenv https://pypi.org/project/pipenv

## Steps after cloning the repository

1. Download the dataset from [here](https://hsluzern-my.sharepoint.com/:u:/g/personal/roman_zberg_stud_hslu_ch/EaHfkzUG6bxAjaDgbnMBjPUBtHMOY3lF5J_W4t7Wg4l4qg?e=e8JH4S) and unzip it in the folder <project-root>/datasets
2. Execute ```pipenv install``` in the project root
3. Download the dataset from: 

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
cd <project-root> 
```

```
pip install -r requirements.txt 
```

```
export PYTHONPATH="$PYTHONPATH:$PWD"
```

```
cd src && python ./train_dino_v2_based_model.py
```


