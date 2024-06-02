import torch
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.outfit_recommendation.utility.MetricsCalculator import MetricsCalculator


@torch.inference_mode()
def get_metrics(
        model,
        dataloader,
        prefix,
        include_roc_curve=True,
        include_conf_matrix=True,
        include_model_certainties=True
):
    model.eval()

    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    predictions = []
    targets = []
    predictions_original = []

    val_acc = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, total=num_batches):
            pred = model(X)

            targets.extend(y.squeeze(1).cpu().to(int).numpy())
            predictions.extend(pred.squeeze(1).round().cpu().to(int).numpy())
            predictions_original.extend(pred.squeeze(1).cpu().to(float).numpy())

            val_acc += (pred.round() == y).type(torch.float).sum().item()

    targets = torch.tensor(targets)
    predictions = torch.tensor(predictions)
    predictions_original = torch.tensor(predictions_original)

    val_acc /= size
    model_f1_score = f1_score(y_true=targets.cpu().numpy(), y_pred=predictions.cpu().numpy())

    ouput_dict = {
        f'{prefix}_f1_score': model_f1_score,
        f'{prefix}_accuracy': val_acc
    }

    if include_conf_matrix:
        ouput_dict.update({
            f'{prefix}_conf_mat': wandb.plot.confusion_matrix(
                y_true=targets.cpu().numpy(),
                preds=predictions.cpu().numpy(),
                class_names=['bad outfit', 'good outfit']
            )
        })

    if include_roc_curve:
        ouput_dict.update({
            f'{prefix}_roc_curve': wandb.plot.roc_curve(
                y_true=targets.cpu().numpy(),
                y_probas=list(map(lambda x: [1 - x, x], predictions_original.cpu().numpy())),
                labels=['bad outfit', 'good outfit']
            )
        })

    if include_model_certainties:
        ouput_dict.update({
            **get_certainties_dict(
                targets=targets,
                predictions=predictions_original,
                prefix=prefix
            )
        })

    return ouput_dict


def get_certainties_dict(
        targets,
        predictions,
        prefix
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
            f'{prefix}_mean_certainty_of_predictions'
        ] = best_model_mean_certainty_of_predictions.item()

    if best_model_mean_certainty_of_correct_predictions is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_correct_predictions'
        ] = best_model_mean_certainty_of_correct_predictions.item()

    if best_model_mean_certainty_of_wrong_predictions is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_wrong_predictions'
        ] = best_model_mean_certainty_of_wrong_predictions.item()

    if best_model_mean_certainty_of_correctly_predicted_good_outfits is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_correctly_predicted_good_outfits'
        ] = best_model_mean_certainty_of_correctly_predicted_good_outfits.item()

    if best_model_mean_certainty_of_correctly_predicted_bad_outfits is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_correctly_predicted_bad_outfits'
        ] = best_model_mean_certainty_of_correctly_predicted_bad_outfits.item()

    if best_model_mean_certainty_of_wrong_predicted_good_outfits is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_wrong_predicted_good_outfits'
        ] = best_model_mean_certainty_of_wrong_predicted_good_outfits.item()

    if best_model_mean_certainty_of_wrong_predicted_bad_outfits is not None:
        certainties_dict[
            f'{prefix}_mean_certainty_of_wrong_predicted_bad_outfits'
        ] = best_model_mean_certainty_of_wrong_predicted_bad_outfits.item()

    return certainties_dict
