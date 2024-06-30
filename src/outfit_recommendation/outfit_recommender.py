import os

from src.clothing_item_types import WearType
from src.outfit_recommendation.data_augmentation import get_data_augmentation_transforms
from src.outfit_recommendation.models.dino_v2_based import OutfitClassifier
import torch
from torchvision.io import read_image
import pandas as pd
from matplotlib import pyplot as plt, gridspec
import numpy as np


class OutfitRecommender:

    def __init__(self, img_root_folder_path, device):
        self._img_root_folder_path = img_root_folder_path
        self._batch_size = 32
        self._tfms = get_data_augmentation_transforms()['test']
        self._device = device

    def _get_potential_outfits(self, clothing_items_df, number_of_potential_outfits, random_state=123):
        inner_wear = clothing_items_df[clothing_items_df['outfit_type'] == WearType.innerWear]

        bottom_wear = clothing_items_df[clothing_items_df['outfit_type'] == WearType.bottomWear]
        shoes = clothing_items_df[clothing_items_df['outfit_type'] == WearType.shoes]
        accessoires = clothing_items_df[clothing_items_df['outfit_type'] == WearType.accessoire]
        outer_wear = clothing_items_df[clothing_items_df['outfit_type'] == WearType.outerWear]

        potential_outfit_df = pd.DataFrame(data={
            'Innerwear_imagePath': list(
                inner_wear.sample(n=number_of_potential_outfits, random_state=random_state)['img_src']),
            'Bottomwear_imagePath': list(
                bottom_wear.sample(n=number_of_potential_outfits, random_state=random_state)['img_src']),
            'Shoes_imagePath': list(shoes.sample(n=number_of_potential_outfits, random_state=random_state)['img_src']),
            'Accessoire_imagePath': list(
                accessoires.sample(n=number_of_potential_outfits, random_state=random_state)['img_src']),
            'Outerwear_imagePath': list(
                outer_wear.sample(n=number_of_potential_outfits, random_state=random_state)['img_src'])
        })

        return potential_outfit_df

    def get_good_outfits(self, model, clothing_items_df, n, save_outfits_to_folder_path=None, random_state=123):
        potential_outfits_df = self._get_potential_outfits(clothing_items_df, n, random_state)

        self.classify_outfits(
            potential_outfits_df, model,
            save_classifications_to_folder_path=save_outfits_to_folder_path
        )

        pass

    def classify_outfits(self, potential_outfit_dataframe: pd.DataFrame, model: OutfitClassifier,
                         save_classifications_to_folder_path=None):

        batch_outfits = torch.tensor([]).to(self._device)
        outfit_counter = 0
        predictions = []
        for index, outfit in potential_outfit_dataframe.iterrows():
            img_accessoire = self._tfms(self._get_image(outfit['Accessoire_imagePath']).to(self._device))
            img_innerwear = self._tfms(self._get_image(outfit['Innerwear_imagePath']).to(self._device))
            img_bottomwear = self._tfms(self._get_image(outfit['Bottomwear_imagePath']).to(self._device))
            img_shoe = self._tfms(self._get_image(outfit['Shoes_imagePath']).to(self._device))
            img_outerwear = self._tfms(self._get_image(outfit['Outerwear_imagePath']).to(self._device))

            batch_outfits = torch.cat([
                batch_outfits,
                torch.cat([
                    img_accessoire.unsqueeze(0),
                    img_innerwear.unsqueeze(0),
                    img_bottomwear.unsqueeze(0),
                    img_shoe.unsqueeze(0),
                    img_outerwear.unsqueeze(0)
                ]).unsqueeze(0)
            ])

            outfit_counter += 1

            if (
                    (outfit_counter % self._batch_size) == 0) or (
                    (
                            outfit_counter == potential_outfit_dataframe.shape[0]
                    )
            ):
                with torch.no_grad():
                    current_n_predictions = len(predictions)
                    original_predictions = model(batch_outfits).cpu().squeeze(1).numpy()
                    predictions.extend(original_predictions)
                if save_classifications_to_folder_path is not None:
                    self.show_batch(
                        batch_outfits,
                        original_predictions >= 0.5,
                        predictions_probas=original_predictions,
                        save_to_folder=save_classifications_to_folder_path,
                        file_names_start=current_n_predictions
                    )
                batch_outfits = torch.tensor([]).to(self._device)

        return predictions

    def _get_image(self, img_path):
        if img_path is not None:
            img_path = img_path.replace('raw/images', 'resized/256x256')

        try:
            if img_path is None:
                return torch.zeros(3, 224, 224)

            else:
                return read_image(f'{self._img_root_folder_path}/{img_path}')
        except Exception as e:
            print(f'Error with image with image path {img_path}')
            raise e

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
    def show_batch(batch, classifications=None, predictions_probas=None, save_to_folder=None, file_names_start=0):
        batch_features = batch

        number_of_outfits = batch_features.shape[0]
        number_of_clothing_items = batch_features.shape[1]

        for batch_item_index in range(number_of_outfits):
            print(batch_item_index)

            added_axes = 0

            if classifications is not None:
                added_axes += 1

            if predictions_probas is not None:
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

            clothing_item_accessoire = OutfitRecommender._get_image_for_matplot_lib(
                batch_features[batch_item_index][0]
            )
            clothing_item_inner_wear = OutfitRecommender._get_image_for_matplot_lib(
                batch_features[batch_item_index][1]
            )
            clothing_item_bottom_wear = OutfitRecommender._get_image_for_matplot_lib(
                batch_features[batch_item_index][2]
            )
            clothing_item_shoes = OutfitRecommender._get_image_for_matplot_lib(
                batch_features[batch_item_index][3]
            )
            clothing_item_over_wear = OutfitRecommender._get_image_for_matplot_lib(
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

            # is_a_good_outfit = batch_target_variables[batch_item_index] == 1
            label_font_size = 20

            if predictions_probas is not None:
                cloting_item_axis_index += 1
                ax = axarr[cloting_item_axis_index]
                ax.text(0.5, 0.5, 'Good Outfit %\n', horizontalalignment='center', transform=ax.transAxes,
                        weight='bold', color='white', fontsize=label_font_size)
                ax.text(0.5, 0.5 - 0.05, round(predictions_probas[batch_item_index], 2), horizontalalignment='center',
                        transform=ax.transAxes,
                        weight='bold', color='white', fontsize=label_font_size)

            if classifications is not None:
                cloting_item_axis_index += 1
                ax = axarr[cloting_item_axis_index]
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
                os.makedirs(save_to_folder, exist_ok=True)

                f.savefig(f'{save_to_folder}/{file_names_start + batch_item_index}')

            plt.show(f)
            plt.close(f)
