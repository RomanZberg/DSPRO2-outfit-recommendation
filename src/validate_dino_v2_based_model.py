import pandas as pd
from src.outfit_recommendation.datasets.image_based_datasets import PolyvoreOutfitDataset
from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier
import torch

from src.train_dino_v2_based_model import get_data_augmentation_transforms


def main(
        pth_file,
        testing_dataset_path,
        zalando_dataset_path,
        dataset_folder_root_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OutfitClassifier.create_from_pth_file(pth_file, device).eval()

    test_dataset_df = pd.read_parquet(testing_dataset_path)
    zalando_dataset_df = pd.read_parquet(zalando_dataset_path)

    test_dataset_path = PolyvoreOutfitDataset(
        testing_dataset_path, get_data_augmentation_transforms()['test'], 'test', 'zero_matrix',
        dataset_folder_root_path, device
    )

    pass


if __name__ == "__main__":
    main(
        pth_file='../trained_models/best_model_median_threshold.pth',
        testing_dataset_path='../datasets/imageBasedModel/polyvore/median_threshold_eb26e630100b98397deda54fa4a0bb95929479bc30e83cbfa72424b7c1e6e178/polyvore_test_6b046cdf467634343bb4fdd8fbdbe02a3746645c7c9108242228da408c97f435.parquet',
        zalando_dataset_path='',
        dataset_folder_root_path='../datasets'
    )
