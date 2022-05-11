from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn import metrics
from sklearn.model_selection import cross_validate
from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        use_scaler: bool,
        max_iter: int,
        logreg_c: float,
) -> None:
    features, target = get_dataset(dataset_path)
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)

        scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
                   'f1': metrics.make_scorer(metrics.f1_score, average='macro'),
                   'recall': metrics.make_scorer(metrics.recall_score, average='macro'),
                   'precision': metrics.make_scorer(metrics.precision_score, average='macro')}

        scores = cross_validate(pipeline, features, target, cv=5, scoring=scoring, return_train_score=False)
        accuracy = scores['test_accuracy'].mean()
        f1 = scores['test_f1'].mean()
        recall = scores['test_recall'].mean()
        precision = scores['test_precision'].mean()

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)

        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"F1: {f1}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"Precision: {precision}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
