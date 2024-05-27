from abc import ABC

import torch


class MetricsCalculator(ABC):

    @staticmethod
    def get_mean_certainty_of_predictions(y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was about its predictions from 0 to 1
        """

        if len(y_predict) == 0:
            return None

        return (y_predict - 0.5).abs().mean() * 2

    @staticmethod
    def get_mean_certainty_of_correct_predictions(y_true: torch.Tensor, y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was about its predictions which were actually true from 0 to 1
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        return (
                       torch.index_select(
                           y_predict, 0,
                           (y_true == (y_predict >= 0.5)).nonzero().squeeze()) - 0.5
               ).abs().mean() * 2

    @staticmethod
    def get_mean_certainty_of_wrong_predictions(y_true: torch.Tensor, y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was about its predictions which were actually wrong from 0 to 1
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        return (
                       torch.index_select(
                           y_predict, 0,
                           (y_true != (y_predict >= 0.5)).nonzero().squeeze()) - 0.5
               ).abs().mean() * 2

    @staticmethod
    def get_mean_certainty_of_correctly_predicted_good_outfits(y_true: torch.Tensor,
                                                               y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was about its predictions of good outfits which were actually good
        outfits
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        y_true_good_outfits = torch.index_select(
            y_true, 0,
            (y_true == 1).nonzero().squeeze()
        )
        predictions_for_good_outfits = torch.index_select(
            y_predict, 0,
            (y_true == 1).nonzero().squeeze()
        )

        return MetricsCalculator.get_mean_certainty_of_correct_predictions(
            y_true_good_outfits,
            predictions_for_good_outfits
        )

    @staticmethod
    def get_mean_certainty_of_correctly_predicted_bad_outfits(y_true: torch.Tensor,
                                                              y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was about its predictions of bad outfits which were actually bad
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        y_true_bad_outfits = torch.index_select(
            y_true, 0,
            (y_true == 0).nonzero().squeeze()
        )
        predictions_for_bad_outfits = torch.index_select(
            y_predict, 0,
            (y_true == 0).nonzero().squeeze()
        )

        return MetricsCalculator.get_mean_certainty_of_correct_predictions(
            y_true_bad_outfits,
            predictions_for_bad_outfits
        )

    @staticmethod
    def get_mean_certainty_of_wrong_predicted_good_outfits(y_true: torch.Tensor,
                                                           y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was when predicting a good outfit when it was actually bad outfit
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        y_true_bad_outfits = torch.index_select(
            y_true, 0,
            (y_true == 0).nonzero().squeeze()
        )

        predictions_for_good_outfits = torch.index_select(
            y_predict, 0,
            (y_true == 0).nonzero().squeeze()
        )

        return MetricsCalculator.get_mean_certainty_of_wrong_predictions(
            y_true_bad_outfits,
            predictions_for_good_outfits
        )

    @staticmethod
    def get_mean_certainty_of_wrong_predicted_bad_outfits(y_true: torch.Tensor,
                                                          y_predict: torch.Tensor) -> torch.Tensor | None:
        """
        Returns the mean of how sure the model was when predicting a bad outfit when it was actually a good outfit
        """

        if (len(y_true) == 0) or (len(y_predict) == 0):
            return None

        y_true_bad_outfits = torch.index_select(
            y_true, 0,
            (y_true == 1).nonzero().squeeze()
        )

        predictions_for_bad_outfits = torch.index_select(
            y_predict, 0,
            (y_true == 1).nonzero().squeeze()
        )

        return MetricsCalculator.get_mean_certainty_of_wrong_predictions(
            y_true_bad_outfits,
            predictions_for_bad_outfits
        )
