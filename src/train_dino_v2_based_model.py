import ssl

import pandas as pd
import torch
import wandb
import optuna

from src.outfit_recommendation.data_augmentation import get_data_augmentation_transforms
from src.outfit_recommendation.datasets.image_based_datasets import PolyvoreOutfitDataset
from src.outfit_recommendation.metrics import get_certainties_dict, get_metrics
from src.outfit_recommendation.utility.EarlyStopper import EarlyStopper

ssl._create_default_https_context = ssl._create_unverified_context
import os
from torch import nn
from torchvision.transforms import v2
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier
from sklearn.metrics import f1_score


def fix_random_seeds(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    targets = []
    predictions = []

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    running_loss = 0.0
    running_corrects = 0

    for batch, (X, y) in tqdm(enumerate(dataloader), desc='Training', total=num_batches):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # saving targets and predictions for later to calculate f1 score
        targets.extend(y.squeeze(1).cpu().to(int).numpy())
        predictions.extend(pred.squeeze(1).round().cpu().to(int).numpy())

        # Statistics
        running_loss += loss.item()
        running_corrects += (pred.round() == y).type(torch.float).sum().item()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / size
    epoch_f1_score = f1_score(y_true=targets, y_pred=predictions)

    return epoch_acc, epoch_f1_score, epoch_loss


@torch.inference_mode()
def val_loop(dataloader, model, loss_fn):
    model.eval()

    targets = []
    predictions = []

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, val_acc = 0.0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Validation', total=num_batches):
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.round() == y).type(torch.float).sum().item()

            # saving targets and predictions for later to calculate f1 score
            targets.extend(y.squeeze(1).cpu().to(int).numpy())
            predictions.extend(pred.squeeze(1).round().cpu().to(int).numpy())

    val_loss /= num_batches
    val_acc /= size

    epoch_f1_score = f1_score(y_true=targets, y_pred=predictions)

    return val_acc, epoch_f1_score, val_loss


@torch.inference_mode()
def get_metrics_for_best_model(model, dataloader):
    model.eval()

    return get_metrics(model, dataloader, 'best_model'),


def train_model(config, data_transforms, dataloaders, device):
    config['data_augmentation'] = data_transforms

    wandb.init(
        # set the wandb project where this run will be logged
        project="ReWear - Outfit Recommender (DSPRO2) Dino v2 based",
        # track hyperparameters and run metadata
        config=config
    )

    outfit_classifier = OutfitClassifier(
        wandb.config['dino_architecture'],
        wandb.config['drop_out'],
        wandb.config['number_of_layers'],
        wandb.config['hidden_neuron_count'],
        device
    )

    optimizer = torch.optim.Adam(
        outfit_classifier.parameters(),
        lr=wandb.config['init_lr']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, wandb.config['epochs'], eta_min=0)

    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=5, min_delta=0)

    best_acc = 0.0
    best_acc_loss = np.inf
    for t in range(wandb.config['epochs']):
        print(f'Epoch {t + 1}\n-------------------------------')

        train_acc, train_f1_score, train_loss = train_loop(dataloaders['train'], outfit_classifier, loss_fn, optimizer)
        scheduler.step()

        val_acc, val_f1_score, val_loss = val_loop(dataloaders['val'], outfit_classifier, loss_fn)

        wandb.log(
            {
                'epoch': t + 1,
                'lr': optimizer.param_groups[0]["lr"],
                'training_accuracy': train_acc,
                'training_f1_score': train_f1_score,
                'training_loss': train_loss,
                'validation_accuracy': val_acc,
                'validation_loss': val_loss,
                'validation_f1_score': val_f1_score,
            }
        )

        if (val_acc == best_acc and val_loss < best_acc_loss) or (val_acc > best_acc):
            best_acc, best_acc_loss = val_acc, val_loss
            save_dict = {
                'epoch': t + 1,
                'state_dict': outfit_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_acc_loss,
                'hyper_parameters': wandb.config
            }
            torch.save(save_dict, os.path.join(wandb.run.dir, 'dino_classifier_ckpt.pth'))

        print('\n')

        if early_stopper.early_stop(val_loss):
            print('Early stopping')
            break

    print('Training completed.')

    # generate metrics for best model
    print('')
    print('Generating metrics for best model')
    checkpoint = torch.load(os.path.join(os.path.join(wandb.run.dir, 'dino_classifier_ckpt.pth')))
    best_config = checkpoint['hyper_parameters']

    model_inf = OutfitClassifier(
        best_config['dino_architecture'],
        best_config['drop_out'],
        best_config['number_of_layers'],
        best_config['hidden_neuron_count'],
        device
    )

    model_inf = model_inf.to(device)

    model_inf.load_state_dict(checkpoint['state_dict'])
    model_inf.eval()

    wandb.log({
        **get_metrics_for_best_model(model_inf, dataloaders['val'])
    })

    wandb.save('dino_classifier_ckpt.pth')
    wandb.finish()

    return best_acc


def main(
        epochs,
        batch_size,
        datasets_folder_root_path,
        empty_image_representation,
        training_dataset_path,
        validation_dataset_path,
        testing_dataset_path,
        seed,
        debug=False
):
    fix_random_seeds(seed)
    data_transforms = get_data_augmentation_transforms()

    # Set a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_df = pd.read_parquet(
        training_dataset_path,
    )

    validation_df = pd.read_parquet(
        validation_dataset_path
    )

    if debug:
        training_df = training_df.head(100)
        validation_df = validation_df.head(100)

    image_datasets = {
        'train': PolyvoreOutfitDataset(training_df, data_transforms['train'], 'training',
                                       empty_image_representation, datasets_folder_root_path, device),
        'val': PolyvoreOutfitDataset(validation_df, data_transforms['val'], 'validation',
                                     empty_image_representation, datasets_folder_root_path, device),
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    def objective(trial: optuna.Trial):
        config = {
            'dino_architecture': 'small',
            'batch_size': batch_size,
            'drop_out': trial.suggest_categorical("hidden_neuron_count", [0.1, 0.2, 0.3]),
            'number_of_layers': trial.suggest_categorical("number_of_layers", range(2, 6)),
            'hidden_neuron_count': trial.suggest_categorical("hidden_neuron_count", [64, 128, 256, 512]),
            'init_lr': 10 ** (-4),
            'epochs': epochs,
            'empty_image_representation': empty_image_representation,
            'training_dataset_path': training_dataset_path,
            'testing_dataset_path': testing_dataset_path,
            'random_seed': seed,
            'debug': debug
        }

        return train_model(config, data_transforms, dataloaders, device)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)


if __name__ == "__main__":
    main(
        epochs=25,
        batch_size=32,
        datasets_folder_root_path='../datasets',
        empty_image_representation='zero_matrix',  # zero_matrix, torch_empty
        training_dataset_path='../datasets/imageBasedModel/polyvore'
                              '/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178'
                              '/polyvore_train_4b80b4128de58ff7338b80300ba8d83cb3b484411b467c99fa5261585a26e9ee'
                              '.parquet',
        validation_dataset_path='../datasets/imageBasedModel/polyvore'
                                '/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178'
                                '/polyvore_validate_f8696e0ecb9cc043aa42bf403e87dd244bbc3c6f101c97e6d84e8aff4c71404e'
                                '.parquet',
        testing_dataset_path='../datasets/imageBasedModel/polyvore'
                             '/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178'
                             '/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
        seed=12345,
        debug=False
    )
