import pandas as pd


# todo: implement gather_data from teradata
def gather_data():
    df_raw = pd.read_csv("data/F1_инф_о_опер.csv", sep=";")
    meta_data = pd.read_table(
        "data/final_meta.csv", sep=",", encoding="cp1251", index_col=0
    )
    raise NotImplementedError
