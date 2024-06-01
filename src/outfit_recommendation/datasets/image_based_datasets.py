import pandas as pd
from matplotlib import pyplot as plt, gridspec
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from tqdm import tqdm
from torchvision.io import read_image

import numpy as np


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

        dataset_index = 0
        self._img_accessoires_no_image_indexes = []
        self._img_innerwear_no_image_indexes = []
        self._img_bottomwear_no_image_indexes = []
        self._img_shoe_no_image_indexes = []
        self._img_outerwear_no_image_indexes = []
        self._valid_outfit_no_image_indexes = []

        for index, outfit in tqdm(self._df.iterrows(), total=self._df.shape[0], desc=f'Loading {name} dataset'):

            if outfit['Accessoire_imagePath'] is None:
                self._img_accessoires_no_image_indexes.append(dataset_index)

            if outfit['Innerwear_imagePath'] is None:
                self._img_innerwear_no_image_indexes.append(dataset_index)

            if outfit['Bottomwear_imagePath'] is None:
                self._img_bottomwear_no_image_indexes.append(dataset_index)

            if outfit['Shoes_imagePath'] is None:
                self._img_shoe_no_image_indexes.append(dataset_index)

            if outfit['Outerwear_imagePath'] is None:
                self._img_outerwear_no_image_indexes.append(dataset_index)

            img_accessoires.append(self._get_image(outfit['Accessoire_imagePath']))
            img_innerwear.append(self._get_image(outfit['Innerwear_imagePath']))
            img_bottomwear.append(self._get_image(outfit['Bottomwear_imagePath']))
            img_shoe.append(self._get_image(outfit['Shoes_imagePath']))
            img_outerwear.append(self._get_image(outfit['Outerwear_imagePath']))
            valid_outfit.append(outfit['valid_outfit'])

            dataset_index += 1

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

        try:
            if img_path is None:
                if self._empty_image_representation == "zero_matrix":
                    return torch.zeros(3, 224, 224)
                elif self._empty_image_representation == "torch_empty":
                    return torch.empty(3, 224, 224)
                else:
                    raise Exception(
                        "Wrong configuration value for key empty_image_representation in model_configuration")
            else:
                return read_image(f'{self._dataset_folder_root_path}/{img_path}')
        except Exception as e:
            print(f'Error with image with image path {img_path}')
            raise e

    def __getitem__(self, index):
        outfit = self.feature_df.iloc[index]
        if index in self._img_accessoires_no_image_indexes:
            img_accessoire = outfit['Accessoire_imagePath'].to(self._device)
        else:
            img_accessoire = self._tfms(outfit['Accessoire_imagePath'].to(self._device))

        if index in self._img_innerwear_no_image_indexes:
            img_innerwear = outfit['Innerwear_imagePath'].to(self._device)
        else:
            img_innerwear = self._tfms(outfit['Innerwear_imagePath'].to(self._device))

        if index in self._img_bottomwear_no_image_indexes:
            img_bottomwear = outfit['Bottomwear_imagePath'].to(self._device)
        else:
            img_bottomwear = self._tfms(outfit['Bottomwear_imagePath'].to(self._device))

        if index in self._img_shoe_no_image_indexes:
            img_shoe = outfit['Shoes_imagePath'].to(self._device)
        else:
            img_shoe = self._tfms(outfit['Shoes_imagePath'].to(self._device))

        if index in self._img_outerwear_no_image_indexes:
            img_outerwear = outfit['Outerwear_imagePath'].to(self._device)
        else:
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

    @staticmethod
    def _get_image_for_matplot_lib(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # fig = plt.figure(figsize=(cols * 3, rows * 3))

        img = img.cpu().numpy().transpose((1, 2, 0))
        # img = img.transpose((1, 2, 0))

        img = std * img + mean
        return np.clip(img, 0, 1)

    @staticmethod
    def show_batch(batch, classifications=None, save_to_folder=None):
        batch_features, batch_target_variables = batch

        number_of_outfits = batch_features.shape[0]
        number_of_clothing_items = batch_features.shape[1]

        for batch_item_index in range(number_of_outfits):

            added_axes = 1

            if classifications is not None:
                added_axes += 1

            f = plt.figure(figsize=(15, 2))

            # Create a GridSpec with 5 columns for the first five axes and 2 columns for the last two axes
            gs = gridspec.GridSpec(1, 7, figure=f)  # 7 axes, with the last two taking up 4 spaces (2 each)

            # Create the first five axes (each one column wide)
            axarr = [f.add_subplot(gs[0, i]) for i in range(number_of_clothing_items + added_axes)]

            # Create the last two axes (each two columns wide)
            # axarr.append(f.add_subplot(gs[0, 5:7]))  # Second to last axis
            # axarr.append(f.add_subplot(gs[0, 7:]))  # Last axis

            f.patch.set_facecolor('black')

            clothing_item_accessoire = PolyvoreOutfitDataset._get_image_for_matplot_lib(
                batch_features[batch_item_index][0]
            )
            clothing_item_inner_wear = PolyvoreOutfitDataset._get_image_for_matplot_lib(
                batch_features[batch_item_index][1]
            )
            clothing_item_bottom_wear = PolyvoreOutfitDataset._get_image_for_matplot_lib(
                batch_features[batch_item_index][2]
            )
            clothing_item_shoes = PolyvoreOutfitDataset._get_image_for_matplot_lib(
                batch_features[batch_item_index][3]
            )
            clothing_item_over_wear = PolyvoreOutfitDataset._get_image_for_matplot_lib(
                batch_features[batch_item_index][4]
            )

            clothing_items = [
                clothing_item_accessoire, clothing_item_over_wear, clothing_item_inner_wear,
                clothing_item_bottom_wear, clothing_item_shoes
            ]

            for ax in axarr:
                ax.set_facecolor("black")
                ax.axis('off')

            for cloting_item_axis_index, clothing_item_image in enumerate(clothing_items):
                ax = axarr[cloting_item_axis_index]
                ax.set_facecolor("black")
                ax.imshow(clothing_item_image)
                ax.axis('off')

            is_a_good_outfit = batch_target_variables[batch_item_index] == 1
            label_font_size = 20

            ax = axarr[cloting_item_axis_index + 1]
            text = 'Actual:\n' if classifications is not None else 'Target\n'
            if is_a_good_outfit:
                ax.text(0.5, 0.5, text, horizontalalignment='center', transform=ax.transAxes,
                        weight='bold', color='white', fontsize=label_font_size)
                ax.text(0.5, 0.5 - 0.05, 'good', horizontalalignment='center', transform=ax.transAxes,
                        weight='bold', color='green', fontsize=label_font_size)
            else:
                ax.text(0.5, 0.5, text, horizontalalignment='center', transform=ax.transAxes,
                        weight='bold', color='white', fontsize=label_font_size)
                ax.text(0.5, 0.5 - 0.05, 'bad', horizontalalignment='center', transform=ax.transAxes,
                        weight='bold', color='red', fontsize=label_font_size)

            if classifications is not None:
                ax = axarr[cloting_item_axis_index + 2]
                text = 'Predicted\n'
                if classifications[batch_item_index] == 1:
                    ax.text(0.5, 0.5, text, horizontalalignment='center', transform=ax.transAxes,
                            weight='bold', color='white', fontsize=label_font_size)
                    ax.text(0.5, 0.5 - 0.05, 'good', horizontalalignment='center', transform=ax.transAxes,
                            weight='bold', color='green', fontsize=label_font_size)
                else:
                    ax.text(0.5, 0.5, text, horizontalalignment='center', transform=ax.transAxes,
                            weight='bold', color='white', fontsize=label_font_size)
                    ax.text(0.5, 0.5 - 0.05, 'bad', horizontalalignment='center', transform=ax.transAxes,
                            weight='bold', color='red', fontsize=label_font_size)

                # ax.axis('off')
            f.tight_layout()

            if save_to_folder is not None:
                f.savefig(save_to_folder)

            plt.show(f)
            plt.close(f)
