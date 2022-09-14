from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer


def create_pipeline(
        model_type: str,
        tuning: bool,
        use_scaler: bool,
        scaler_type: str,
        max_iter: int,
        logreg_C: float,
        feature_selection: bool,
        random_state: int,
        n_estimators: int,
        max_depth: int,
        criterion: str,
        min_samples_leaf: int,
        bootstrap: bool
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        if scaler_type == 'MinMax':
            pipeline_steps.append(("scaler", MinMaxScaler()))
        elif scaler_type == 'Standard':
            pipeline_steps.append(("scaler", StandardScaler()))
        else:
            pipeline_steps.append(
                (
                    "scaler",
                    ColumnTransformer(
                        transformers=[("scaler", StandardScaler(), get_num_columns())],
                        remainder="passthrough",
                    ),
                )
            )
    if feature_selection:
        pipeline_steps.append(
            (
                "feature_selection",
                SelectFromModel(RandomForestClassifier(random_state=2022)),
            )
        )
    if model_type == 'RandomForest':
        if tuning:
            parametrs = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [8, 10, 12, 15, 20, None],
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
                                           random_state=random_state, min_samples_leaf=min_samples_leaf,
                                           bootstrap=bootstrap),
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


def get_num_columns() -> list[str]:
    return [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]