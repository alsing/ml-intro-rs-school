from pandas_profiling import ProfileReport
from .data import get_dataset
import pandas as pd
import numpy as np
import click
from pathlib import Path


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the dataset",
    show_default=True,
)
@click.option(
    "-s",
    "--save-report-path",
    default="data/eda.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the report file",
    show_default=True,
)
def eda(dataset_path: Path, save_report_path: Path) -> None:
    features, target = get_dataset(dataset_path)
    profile = ProfileReport(
        pd.concat([features, target], axis=1), title="Pandas Profiling Report"
    )
    profile.to_file(save_report_path)


def preprocess_data(data: pd.DataFrame):
    """soil1 = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type6', 'Soil_Type10', 'Soil_Type11',
             'Soil_Type12', 'Soil_Type13', 'Soil_Type17', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type29',
             'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    soil2 = ['Soil_Type5', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
             'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
             'Soil_Type28', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37']
    data['Soil_Type'] = data[soil1].sum(axis=1) / 21
    data = data.drop(soil1 + soil2, axis=1)
    return data"""
    data["Binned_Elevation"] = data["Elevation"].apply(lambda x: np.floor(x / 50))
    data["Horizontal_Distance_To_Roadways_Log"] = data[
        "Horizontal_Distance_To_Roadways"
    ].apply(lambda x: np.log(x, where=(x != 0)))
    data["Horizontal_Distance_To_Fire_Points_Log"] = data[
        "Horizontal_Distance_To_Fire_Points"
    ].apply(lambda x: np.log(x, where=(x != 0)))
    data["Horizontal_Distance_To_Hydrology_Log"] = data[
        "Horizontal_Distance_To_Hydrology"
    ].apply(lambda x: np.log(x, where=(x != 0)))
    return data
