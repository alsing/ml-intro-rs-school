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
    "--model-type",
    default='LogReg',
    type=str,
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
    "--scaler-type",
    default='Standard',
    type=str,
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
@click.option(
    "--apply-pca",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=75,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=8,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default='entropy',
    type=str,
    show_default=True,
)
@click.option(
    "--bootstrap",
    default=True,
    type=bool,
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        model_type: str,
        random_state: int,
        use_scaler: bool,
        scaler_type: str,
        max_iter: int,
        logreg_c: float,
        apply_pca: bool,
        n_estimators: int,
        max_depth: int,
        criterion: str,
        bootstrap: bool
) -> None:
    features, target = get_dataset(dataset_path)
    with mlflow.start_run():
        pipeline = create_pipeline(model_type, use_scaler, scaler_type, max_iter, logreg_c, apply_pca, random_state,
                                   n_estimators, max_depth, criterion, bootstrap)

        scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
                   'f1': metrics.make_scorer(metrics.f1_score, average='macro'),
                   'recall': metrics.make_scorer(metrics.recall_score, average='macro'),
                   'precision': metrics.make_scorer(metrics.precision_score, average='macro')}

        scores = cross_validate(pipeline, features, target, cv=5, scoring=scoring, return_train_score=False)
        accuracy = scores['test_accuracy'].mean()
        f1 = scores['test_f1'].mean()
        recall = scores['test_recall'].mean()
        precision = scores['test_precision'].mean()

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("scaler_type", scaler_type)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("apply_pca", apply_pca)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("bootstrap", bootstrap)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"F1: {f1}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"Precision: {precision}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
