from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def create_pipeline(
    use_scaler: bool, scaler_type: str, max_iter: int, logreg_C: float, apply_pca: bool, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        if scaler_type == 'MinMax':
            pipeline_steps.append(("scaler", MinMaxScaler()))
        else:
            pipeline_steps.append(("scaler", StandardScaler()))
    if apply_pca:
        pipeline_steps.append(('pca', PCA()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
