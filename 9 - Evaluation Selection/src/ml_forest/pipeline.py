from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def create_pipeline(
        model_type: str,
        tuning: bool,
        use_scaler: bool,
        scaler_type: str,
        max_iter: int,
        logreg_C: float,
        apply_pca: bool,
        random_state: int,
        n_estimators: int,
        max_depth: int,
        criterion: str,
        bootstrap: bool
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        if scaler_type == 'MinMax':
            pipeline_steps.append(("scaler", MinMaxScaler()))
        else:
            pipeline_steps.append(("scaler", StandardScaler()))
    if apply_pca:
        pipeline_steps.append(('pca', PCA()))
    if model_type == 'RandomForest':
        if tuning:
            parametrs = {
                'n_estimators': [5, 10, 20, 35, 50, 75, 100, 150],
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True, False]
            }
            forest = RandomForestClassifier()
            pipeline_steps.append(
                (
                    "classifier",
                    GridSearchCV(forest, parametrs, cv=5, scoring='accuracy', n_jobs=-1, refit=True),
                )
            )
        else:
            pipeline_steps.append(
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                           bootstrap=bootstrap, random_state=random_state),
                )
            )
    else:
        if tuning:
            parametrs = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [1, 0.1, 0.01, 0.5, 10, 100, 2, 5],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }
            logreg = LogisticRegression()
            pipeline_steps.append(
                (
                    "classifier",
                    GridSearchCV(logreg, parametrs, cv=5, scoring='accuracy', n_jobs=-1, refit=True),
                )
            )
        else:
            pipeline_steps.append(
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state, max_iter=max_iter, C=logreg_C
                    ),
                )
            )
    return Pipeline(steps=pipeline_steps)
