import unittest
import torch

from src.outfit_recommendation.utility.MetricsCalculator import MetricsCalculator


class MetricsCalculatorTest(unittest.TestCase):

    def test_get_mean_certainty_of_predictions(self):
        y_predictions = torch.tensor([0.55, 0, 0.4, 1, 0.1, 0.88, 0.51])

        mean_certainty_of_predictions = MetricsCalculator.get_mean_certainty_of_predictions(y_predictions)

        # certainty's = 0.05, 0.5, 0.1, 0.5, 0.4, 0.38, 0.01
        self.assertEqual(
            torch.tensor(0.554285714),
            mean_certainty_of_predictions
        )

    def test_get_mean_certainty_of_predictions_hundred_percent_certain(self):
        y_predictions = torch.tensor([0, 1, 1, 1, 0, 1, 0])

        mean_certainty_of_predictions = MetricsCalculator.get_mean_certainty_of_predictions(y_predictions)

        self.assertEqual(
            torch.tensor(1),
            mean_certainty_of_predictions
        )

    def test_get_mean_certainty_of_predictions_hundred_percent_uncertain(self):
        y_predictions = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        mean_certainty_of_predictions = MetricsCalculator.get_mean_certainty_of_predictions(y_predictions)

        self.assertEqual(
            torch.tensor(0),
            mean_certainty_of_predictions
        )

    def test_get_mean_certainty_of_correct_predictions(self):
        y_true = torch.tensor([1, 0, 0, 1, 0, 1, 1])
        y_predictions = torch.tensor([0.55, 0.53, 0.4, 1, 0.1, 0.49, 0.51])

        mean_certainty_of_correct_predictions = MetricsCalculator.get_mean_certainty_of_correct_predictions(
            y_true,
            y_predictions
        )

        # correct certainty's = 0.05, 0.1, 0.5, 0.4, 0.01
        expected = torch.tensor(0.424)
        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_correct_predictions, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_correct_predictions}'
        )

    def test_get_mean_certainty_of_wrong_predictions(self):
        y_true = torch.tensor([1, 0, 0, 1, 0, 1, 1])
        y_predictions = torch.tensor([0.55, 0.53, 0.4, 1, 0.1, 0.49, 0.51])

        mean_certainty_of_wrong_predictions = MetricsCalculator.get_mean_certainty_of_wrong_predictions(
            y_true,
            y_predictions
        )

        # wrong certainty's = 0.03, 0.01
        expected = torch.tensor(0.04)
        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_wrong_predictions, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_wrong_predictions}'
        )

    def test_get_mean_certainty_of_correctly_predicted_good_outfits(self):
        y_true = torch.tensor(      [1,        0,   0, 1,   0,    1,    1])
        y_predictions = torch.tensor([0.55, 0.53, 0.4, 1, 0.1, 0.49, 0.51])

        mean_certainty_of_correctly_predicted_good_outfits = MetricsCalculator.get_mean_certainty_of_correctly_predicted_good_outfits(
            y_true, y_predictions)

        # correct certainty's = 0.05, 0.5, 0.01
        expected = torch.tensor(0.373333333)
        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_correctly_predicted_good_outfits, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_correctly_predicted_good_outfits}'
        )

    def test_get_mean_certainty_of_correctly_predicted_bad_outfits(self):
        y_true = torch.tensor(       [1,       0,   0, 1,   0,    1,    1])
        y_predictions = torch.tensor([0.55, 0.53, 0.4, 1, 0.1, 0.49, 0.51])

        mean_certainty_of_correctly_predicted_bad_outfits = MetricsCalculator.get_mean_certainty_of_correctly_predicted_bad_outfits(
            y_true, y_predictions)

        # correct certainty's = 0.1, 0.4
        expected = torch.tensor(0.5)

        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_correctly_predicted_bad_outfits, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_correctly_predicted_bad_outfits}'
        )

    def test_get_mean_certainty_of_wrong_predicted_good_outfits(self):
        y_true = torch.tensor(      [0,        0,   0,    1,   0,    1,    0])
        y_predictions = torch.tensor([0.55, 0.53, 0.4, 0.32, 0.1, 0.49, 0.51])

        mean_certainty_of_wrong_predicted_good_outfits = MetricsCalculator.get_mean_certainty_of_wrong_predicted_good_outfits(
            y_true, y_predictions)

        # wrong certainty's = 0.05, 0.03, 0.01
        expected = torch.tensor(0.06)

        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_wrong_predicted_good_outfits, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_wrong_predicted_good_outfits}'
        )

    def test_get_mean_certainty_of_wrong_predicted_bad_outfits(self):
        y_true = torch.tensor(      [1,        0,   0, 1,   0,   1,     0])
        y_predictions = torch.tensor([0.38, 0.53, 0.4, 1, 0.1, 0.48, 0.51])

        mean_certainty_of_wrong_predicted_bad_outfits = MetricsCalculator.get_mean_certainty_of_wrong_predicted_bad_outfits(
            y_true, y_predictions)

        # correct certainty's = 0.12, 0.02
        expected = torch.tensor(0.14)


        self.assertTrue(
            torch.isclose(expected, mean_certainty_of_wrong_predicted_bad_outfits, atol=1e-8).all(),
            f'Value should be {expected} but was {mean_certainty_of_wrong_predicted_bad_outfits}'
        )

