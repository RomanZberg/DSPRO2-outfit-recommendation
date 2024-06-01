import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.outfit_recommendation.datasets.image_based_datasets import PolyvoreOutfitDataset
from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier
import torch
from sklearn.metrics import f1_score
from src.train_dino_v2_based_model import get_data_augmentation_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def get_metrics(model, dataloader, prefix):
    model.eval()

    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    predictions = []
    targets = []
    predictions_original = []

    val_acc = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Validation', total=num_batches):
            pred = model(X)

            targets.extend(y.squeeze(1).cpu().to(int).numpy())
            predictions.extend(pred.squeeze(1).round().cpu().to(int).numpy())
            predictions_original.extend(pred.squeeze(1).cpu().to(float).numpy())
            val_acc += (pred.round() == y).type(torch.float).sum().item()

    val_acc /= size

    targets = torch.tensor(targets)
    predictions = torch.tensor(predictions)
    predictions_original = torch.tensor(predictions_original)

    model_f1_score = f1_score(targets.cpu().numpy(), predictions.cpu().numpy())

    return {
        f'{prefix}_f1_score': model_f1_score,
        f'{prefix}_accuracy': val_acc
        # f'{prefix}_conf_mat': wandb.plot.confusion_matrix(
        #     y_true=targets.cpu().numpy(),
        #     preds=predictions.cpu().numpy(),
        #     class_names=['bad outfit', 'good outfit']
        # )
    }


def validate_zalando_dataset_with_outfits_of_type(
        oufit_type: str, model: OutfitClassifier, zalando_df: pd.DataFrame,
        dataset_folder_root_path
):
    if not oufit_type == "all":
        zalando_df = zalando_df[zalando_df['outfit_type'] == oufit_type]

        if zalando_df.shape[0] == 0:
            raise Exception(f'The requested outfit type {oufit_type} has not entries in the zalando_dataset')

    zalando_dataset = PolyvoreOutfitDataset(
        zalando_df, get_data_augmentation_transforms()['test'],
        f'zalando outfit type {oufit_type}',
        'zero_matrix', dataset_folder_root_path,
        device
    )

    dataloader = DataLoader(
        zalando_dataset, batch_size=32,
        shuffle=True, num_workers=0
    )

    return get_metrics(model, dataloader, f'zalando_dataset_outfit_type_{oufit_type}')


def vaildate_on_zalando_dataset(model: OutfitClassifier, zalando_df: pd.DataFrame, dataset_folder_root_path):
    all_metrics = validate_zalando_dataset_with_outfits_of_type(
        'all', model, zalando_df, dataset_folder_root_path
    )

    casual_metrics = validate_zalando_dataset_with_outfits_of_type(
        'casual', model, zalando_df, dataset_folder_root_path
    )

    urban_metrics = validate_zalando_dataset_with_outfits_of_type(
        'urban', model, zalando_df, dataset_folder_root_path
    )

    sporty_metrics = validate_zalando_dataset_with_outfits_of_type(
        'sporty', model, zalando_df, dataset_folder_root_path
    )

    extravagant_metrics = validate_zalando_dataset_with_outfits_of_type(
        'extravagant', model, zalando_df, dataset_folder_root_path
    )

    party_metrics = validate_zalando_dataset_with_outfits_of_type(
        'party', model, zalando_df, dataset_folder_root_path
    )

    classic_metrics = validate_zalando_dataset_with_outfits_of_type(
        'classic', model, zalando_df, dataset_folder_root_path
    )

    return {
        **all_metrics,
        **casual_metrics,
        **urban_metrics,
        **sporty_metrics,
        **extravagant_metrics,
        **party_metrics,
        **classic_metrics
    }


def vaildate_on_polyvore_dataset(model, polyvore_df: pd.DataFrame, dataset_folder_root_path):
    polyvore_test_dataset = PolyvoreOutfitDataset(
        polyvore_df, get_data_augmentation_transforms()['test'], 'polyvore test', 'zero_matrix',
        dataset_folder_root_path, device
    )

    dataloader = DataLoader(
        polyvore_test_dataset, batch_size=32,
        shuffle=True, num_workers=0
    )

    return get_metrics(model, dataloader, 'polyvore')


def main(
        pth_file,
        testing_dataset_path,
        zalando_dataset_path,
        polyvore_datasets_folder_root_path,
        zalando_datasets_folder_root_path
):
    model = OutfitClassifier.create_from_pth_file(pth_file, device).eval()

    polyvore_test_dataset_df = pd.read_parquet(testing_dataset_path)
    zalando_test_dataset_df = pd.read_parquet(zalando_dataset_path)

    result_dict_polyvore = vaildate_on_polyvore_dataset(
        model, polyvore_test_dataset_df,
        polyvore_datasets_folder_root_path
    )

    result_dict_zalando = vaildate_on_zalando_dataset(
        model, zalando_test_dataset_df, zalando_datasets_folder_root_path
    )

    log_dict = {
        **result_dict_polyvore,
        **result_dict_zalando
    }

    print(log_dict)


if __name__ == "__main__":
    main(
        pth_file='../trained_models/best_model_median_threshold.pth',
        testing_dataset_path='../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
        zalando_dataset_path='../datasets/imageBasedModel/zalando/zalando_ea4bf73acd0119c31a8a708bce2a2b302e72c1ecebc7c2265ee80cff3a15ed0a.parquet',
        polyvore_datasets_folder_root_path='../datasets',
        zalando_datasets_folder_root_path='../datasets/imageBasedModel/zalando'
    )
