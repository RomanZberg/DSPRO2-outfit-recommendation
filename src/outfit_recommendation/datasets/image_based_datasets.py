import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from tqdm import tqdm
from torchvision.io import read_image


class PolyvoreOutfitDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_frame,
            tfms: v2.Compose,
            name,
            empty_image_representation,
            dataset_folder_root_path,
            device
    ):
        self._df = data_frame
        self._tfms = tfms
        self._device = device
        self._empty_image_representation = empty_image_representation
        self._dataset_folder_root_path = dataset_folder_root_path

        img_accessoires = []
        img_innerwear = []
        img_bottomwear = []
        img_shoe = []
        img_outerwear = []
        valid_outfit = []

        for index, outfit in tqdm(self._df.iterrows(), total=self._df.shape[0], desc=f'Loading {name} dataset'):
            img_accessoires.append(self._get_image(outfit['Accessoire_imagePath']))
            img_innerwear.append(self._get_image(outfit['Innerwear_imagePath']))
            img_bottomwear.append(self._get_image(outfit['Bottomwear_imagePath']))
            img_shoe.append(self._get_image(outfit['Shoes_imagePath']))
            img_outerwear.append(self._get_image(outfit['Outerwear_imagePath']))
            valid_outfit.append(outfit['valid_outfit'])

        self.feature_df = pd.DataFrame({
            'Accessoire_imagePath': img_accessoires,
            'Innerwear_imagePath': img_innerwear,
            'Bottomwear_imagePath': img_bottomwear,
            'Shoes_imagePath': img_shoe,
            'Outerwear_imagePath': img_outerwear,
            'valid_outfit': valid_outfit
        }, index=self._df.index)

    # Class for test dataset
    def _get_image(self, img_path):
        if img_path is not None:
            img_path = img_path.replace('raw/images', 'resized/256x256')

        if img_path is None:
            if self._empty_image_representation == "zero_matrix":
                return torch.zeros(3, 224, 224)
            elif self._empty_image_representation == "torch_empty":
                return torch.empty(3, 224, 224)
            else:
                raise Exception("Wrong configuration value for key empty_image_representation in model_configuration")
        else:
            return read_image(f'{self._dataset_folder_root_path}/{img_path}')

    def __getitem__(self, index):
        outfit = self.feature_df.iloc[index]
        img_accessoire = self._tfms(outfit['Accessoire_imagePath'].to(self._device))
        img_innerwear = self._tfms(outfit['Innerwear_imagePath'].to(self._device))
        img_bottomwear = self._tfms(outfit['Bottomwear_imagePath'].to(self._device))
        img_shoe = self._tfms(outfit['Shoes_imagePath'].to(self._device))
        img_outerwear = self._tfms(outfit['Outerwear_imagePath'].to(self._device))

        target_variable = torch.tensor([outfit['valid_outfit']]).to(torch.float).to(self._device)

        feature_tensor = torch.cat([
            img_accessoire.unsqueeze(0),
            img_innerwear.unsqueeze(0),
            img_bottomwear.unsqueeze(0),
            img_shoe.unsqueeze(0),
            img_outerwear.unsqueeze(0)
        ]).to(self._device)

        return feature_tensor, target_variable

    def __len__(self):
        return self._df.shape[0]
