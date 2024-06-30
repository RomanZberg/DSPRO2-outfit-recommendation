import pandas as pd
import wandb
from torch.utils.data import DataLoader

from src.outfit_recommendation.data_augmentation import get_data_augmentation_transforms
from src.outfit_recommendation.datasets.image_based_datasets import PolyvoreOutfitDataset
from src.outfit_recommendation.metrics import get_metrics
from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    return get_metrics(model, dataloader, f'zalando_dataset_outfit_type_{oufit_type}', include_roc_curve=False,
                       include_conf_matrix=False)


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
        pth_files,
        testing_dataset_paths,
        zalando_dataset_path,
        polyvore_datasets_folder_root_path,
        zalando_datasets_folder_root_path
):
    for index, pth_file in enumerate(pth_files):
        wandb.init(
            # set the wandb project where this run will be logged
            project="ReWear - Outfit Recommender (DSPRO2) Dino v2 based testing",
            config={
                'pth_file': pth_file,
                'testing_dataset_path': testing_dataset_paths[index],
                'zalando_dataset_path': zalando_dataset_path
            }
        )

        model = OutfitClassifier.create_from_pth_file(pth_file, device).eval()

        polyvore_test_dataset_df = pd.read_parquet(testing_dataset_paths[index])
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

        wandb.log(
            log_dict
        )

        print(log_dict)

        wandb.finish()


if __name__ == "__main__":
    main(
        pth_files=[
            '../trained_models/volanic-moon-9-median-threshold.pth',
            '../trained_models/honest-grass-25-median-threshold.pth',
            '../trained_models/cosmic-sunset-49-q1-threshold.pth',
            '../trained_models/lively-planet-39-q1-threshold.pth'
        ],
        testing_dataset_paths=[
            '../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
            '../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
            '../datasets/imageBasedModel/polyvore/q1_threshold_oversampled_93bb63f88b5735b75cbac6b4549ca7de6cb6759b70cdd99cb2efaa2cad1ea359/polyvore_test_eccba23b76ec907bfc446b3ff374930b55c2d9609d33fac87bccf5b2fb95db71.parquet',
            '../datasets/imageBasedModel/polyvore/q1_threshold_oversampled_93bb63f88b5735b75cbac6b4549ca7de6cb6759b70cdd99cb2efaa2cad1ea359/polyvore_test_eccba23b76ec907bfc446b3ff374930b55c2d9609d33fac87bccf5b2fb95db71.parquet'
        ],
        zalando_dataset_path='../datasets/imageBasedModel/zalando/zalando_ea4bf73acd0119c31a8a708bce2a2b302e72c1ecebc7c2265ee80cff3a15ed0a.parquet',
        polyvore_datasets_folder_root_path='../datasets',
        zalando_datasets_folder_root_path='../datasets/imageBasedModel/zalando'
    )
