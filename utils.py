import pandas as pd
import dill


def save_dict_to_file(dictionary: dict, filename: str) -> None:
    with open(filename, "wb") as file:
        dill.dump(dictionary, file)


def load_dict_from_file(filename: str):
    with open(filename, "rb") as file:
        dictionary = dill.load(file)
    return dictionary


def save_csv(data: pd.DataFrame, filename: str) -> None:
    data.to_csv(filename, index=False)


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)
