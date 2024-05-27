import ssl

import pandas as pd
import torch
import wandb
import optuna

from src.outfit_recommendation.datasets.image_based_datasets import PolyvoreOutfitDataset
from src.outfit_recommendation.utility.EarlyStopper import EarlyStopper
from src.outfit_recommendation.utility.MetricsCalculator import MetricsCalculator

ssl._create_default_https_context = ssl._create_unverified_context
import os
from torch import nn
from torchvision.transforms import v2
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier


def fix_random_seeds(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_loop(dataloader, feature_model, loss_fn, optimizer):
    feature_model.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0.0
    running_corrects = 0
    feature_model.counter = 0

    for batch, (X, y) in tqdm(enumerate(dataloader), desc='Training', total=num_batches):
        # Compute prediction and loss
        pred = feature_model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        running_corrects += (pred.round() == y).type(torch.float).sum().item()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / size

    return epoch_acc, epoch_loss


@torch.inference_mode()
def val_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, val_acc = 0.0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Validation', total=num_batches):
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.round() == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_acc /= size

    return val_acc, val_loss


def get_data_augmentation_transforms():
    return {
        'train': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.RandomHorizontalFlip(),
            v2.RandomPerspective(fill=255),
            v2.RandomAffine(30, fill=255),
            # v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        'val': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        'test': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
    }


def get_certainties_dict(
        targets,
        predictions
):
    best_model_mean_certainty_of_predictions = MetricsCalculator.get_mean_certainty_of_predictions(
        y_predict=predictions
    )

    best_model_mean_certainty_of_correct_predictions = MetricsCalculator.get_mean_certainty_of_correct_predictions(
        y_true=targets,
        y_predict=predictions
    )

    best_model_mean_certainty_of_wrong_predictions = MetricsCalculator.get_mean_certainty_of_wrong_predictions(
        y_true=targets,
        y_predict=predictions
    )

    best_model_mean_certainty_of_correctly_predicted_good_outfits = MetricsCalculator.get_mean_certainty_of_correctly_predicted_good_outfits(
        y_true=targets,
        y_predict=predictions
    )

    best_model_mean_certainty_of_correctly_predicted_bad_outfits = MetricsCalculator.get_mean_certainty_of_correctly_predicted_bad_outfits(
        y_true=targets,
        y_predict=predictions
    )

    best_model_mean_certainty_of_wrong_predicted_good_outfits = MetricsCalculator.get_mean_certainty_of_wrong_predicted_good_outfits(
        y_true=targets,
        y_predict=predictions
    )

    best_model_mean_certainty_of_wrong_predicted_bad_outfits = MetricsCalculator.get_mean_certainty_of_wrong_predicted_bad_outfits(
        y_true=targets,
        y_predict=predictions
    )

    certainties_dict = {}

    if best_model_mean_certainty_of_predictions is not None:
        certainties_dict[
            'best_model_mean_certainty_of_predictions'
        ] = best_model_mean_certainty_of_predictions.item()

    if best_model_mean_certainty_of_correct_predictions is not None:
        certainties_dict[
            'best_model_mean_certainty_of_correct_predictions'
        ] = best_model_mean_certainty_of_correct_predictions.item()

    if best_model_mean_certainty_of_wrong_predictions is not None:
        certainties_dict[
            'best_model_mean_certainty_of_wrong_predictions'
        ] = best_model_mean_certainty_of_wrong_predictions.item()

    if best_model_mean_certainty_of_correctly_predicted_good_outfits is not None:
        certainties_dict[
            'best_model_mean_certainty_of_correctly_predicted_good_outfits'
        ] = best_model_mean_certainty_of_correctly_predicted_good_outfits.item()

    if best_model_mean_certainty_of_correctly_predicted_bad_outfits is not None:
        certainties_dict[
            'best_model_mean_certainty_of_correctly_predicted_bad_outfits'
        ] = best_model_mean_certainty_of_correctly_predicted_bad_outfits.item()

    if best_model_mean_certainty_of_wrong_predicted_good_outfits is not None:
        certainties_dict[
            'best_model_mean_certainty_of_wrong_predicted_good_outfits'
        ] = best_model_mean_certainty_of_wrong_predicted_good_outfits.item()

    if best_model_mean_certainty_of_wrong_predicted_bad_outfits is not None:
        certainties_dict[
            'best_model_mean_certainty_of_wrong_predicted_bad_outfits'
        ] = best_model_mean_certainty_of_wrong_predicted_bad_outfits.item()

    return certainties_dict


@torch.inference_mode()
def get_metrics(model, dataloader, loss_fn):
    model.eval()

    num_batches = len(dataloader)

    predictions = []
    targets = []
    predictions_original = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Validation', total=num_batches):
            pred = model(X)
            targets.extend(y.squeeze().to(int).numpy())
            predictions.extend(pred.squeeze().round().to(int).numpy())
            predictions_original.extend(pred.squeeze().to(float).numpy())

    targets = torch.tensor(targets)
    predictions = torch.tensor(predictions)
    predictions_original = torch.tensor(predictions_original)

    return {
        'conf_mat': wandb.plot.confusion_matrix(
            y_true=targets.numpy(),
            preds=predictions.numpy(),
            class_names=['bad outfit', 'good outfit']
        ),
        **get_certainties_dict(
            targets=targets,
            predictions=predictions_original
        )
    }


def train_model(config, data_transforms, dataloaders, device):
    outfit_classifier = OutfitClassifier(
        config['dino_architecture'],
        config['drop_out'],
        config['number_of_layers'],
        config['hidden_neuron_count'],
        device
    )

    config['data_augmentation'] = data_transforms

    wandb.init(
        # set the wandb project where this run will be logged
        project="ReWear - Outfit Recommender (DSPRO2) Dino v2 based",
        # track hyperparameters and run metadata
        config=config
    )

    optimizer = torch.optim.Adam(
        outfit_classifier.parameters(),
        lr=config['init_lr']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=0)

    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=5, min_delta=0)

    best_acc = 0.0
    best_acc_loss = np.inf
    for t in range(config['epochs']):
        print(f'Epoch {t + 1}\n-------------------------------')

        train_acc, train_loss = train_loop(dataloaders['train'], outfit_classifier, loss_fn, optimizer)
        scheduler.step()

        val_acc, val_loss = val_loop(dataloaders['val'], outfit_classifier, loss_fn)

        wandb.log(
            {
                'epoch': t,
                'lr': optimizer.param_groups[0]["lr"],
                'training_accuracy': train_acc,
                'training_loss': train_loss,
                'validation_accuracy': val_acc,
                'validation_loss': val_loss
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
                'hyper_parameters': config
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
        **get_metrics(model_inf, dataloaders['val'], loss_fn)
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
        testing_dataset_path,
        debug=False
):
    fix_random_seeds()
    data_transforms = get_data_augmentation_transforms()

    # Set a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_df = pd.read_parquet(
        training_dataset_path,
    )

    if debug:
        training_df = training_df.head(100)

    train, validation = train_test_split(
        training_df, test_size=0.25, random_state=42,
        stratify=training_df['valid_outfit']
    )

    image_datasets = {
        'train': PolyvoreOutfitDataset(train, data_transforms['train'], 'training',
                                       empty_image_representation, datasets_folder_root_path, device),
        'val': PolyvoreOutfitDataset(validation, data_transforms['val'], 'validation',
                                     empty_image_representation, datasets_folder_root_path, device),
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    def objective(trial: optuna.Trial):
        config = {
            'dino_architecture': 'small',
            'batch_size': batch_size,
            'drop_out': 0,
            'number_of_layers': trial.suggest_categorical("number_of_layers", range(2, 6)),
            'hidden_neuron_count': trial.suggest_categorical("hidden_neuron_count", [45, 64, 128, 256]),
            'init_lr': 10 ** (-4),
            'epochs': epochs,
            'empty_image_representation': empty_image_representation,
            'training_dataset_path': training_dataset_path,
            'testing_dataset_path': testing_dataset_path
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
        training_dataset_path='../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_train_d9a4be80b78df9a5ef8b6682f5785becc175d5c3ff17cf7428574485e72c62f8.parquet',
        testing_dataset_path='../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
        debug=False
    )
