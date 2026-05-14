import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import itertools
from tqdm import tqdm
from datetime import datetime, timedelta


def calc_stats_demand(df: pd.DataFrame, TRAIN_START: datetime, qq: float = 0.95) -> defaultdict:
    """
    Function returns dict_1 (key=atm_id, value=dict_2) of dict_2 (key=[stat_k_mean, stat_k_n, stat_sums], value=statistics).
    It will calc prop for each sum for each hour, use statistics for the last 90 days.

    * For demand (DIS)

    Input:
        -    data: pd.DataFrame - filtered train DataFrame to process
            - atm_id         - int
            - dateTime_oper  - DateTime format for each transaction
            - sum            - int
        - qq - quantile, percentage of data in np.array
    Output:
        - {'atm_id': {'stat_k_mean': np.array, 'stat_k_n': np.array, 'stat_sums': np.array}}
    """

    def preprocess_for_stats(df: pd.DataFrame) -> pd.DataFrame:
        def date_filtering(df: pd.DataFrame, train_start, train_end):
            df = df[
                (df["dateTime_oper"] >= train_start)
                & (df["dateTime_oper"] <= train_end)
            ]
            return df

        TRAIN_END = df["dateTime_oper"].max() - timedelta(days=28)

        df = date_filtering(df, TRAIN_START, TRAIN_END)

        df["n_oper"] = 1
        df = df.groupby([df["atm_id"], df["dateTime_oper"].dt.hour, df["day_of_week"]]).agg(
            {"sum": [lambda x: list(x)], "n_oper": "sum"}
        )
        df.reset_index(inplace=True)
        df.columns = columns
        return df

    def get_stat_sums(lst: np.array) -> np.array:
        n: int = lst.shape[0]
        if n != 0:
            lst = lst[lst <= np.quantile(lst, qq)]
            prob = np.zeros((int(lst.max() // 5)) + 1)
            for k, v in Counter(lst).items():
                prob[int(k // 5)] = v / n
        else:
            prob = np.zeros((1))
        return prob

    df['day_of_week'] = df['dateTime_oper'].apply(lambda row: row.weekday())

    output_dict = defaultdict(lambda: defaultdict(np.array))
    columns = ["atm_id", "hour", "day_of_week", "list_sums", "n_oper"]

    df = preprocess_for_stats(df)
    stat_list = df.groupby(["atm_id"])["list_sums"].apply(
        lambda x: np.array(list((itertools.chain.from_iterable(list(x)))))
    )
    stat_sums = stat_list.apply(get_stat_sums)

    for atm_id in tqdm(df['atm_id'].unique()):
        stat_k_mean = np.zeros((7, 24))
        stat_k_n = np.zeros((7, 24))
        for day_of_week in range(7):
            atm_df = df[(df['atm_id'] == atm_id) & (df['day_of_week'] == day_of_week)]
            if atm_df.shape[0] == 0:
                stat_k_mean[day_of_week] = stat_k_mean[0]
                stat_k_n[day_of_week] = stat_k_n[0]
            else:

                atm_df['mean_sum'] = atm_df['list_sums'].apply(lambda x: np.sum(x))
                total_sum = sum(atm_df['list_sums'].sum())
                total_n = sum(atm_df['n_oper'])
                atm_df['k_mean'] = atm_df['mean_sum'] / total_sum
                atm_df['k_n'] = atm_df['n_oper'] / total_n

                idx = atm_df['hour'].to_list()
                values_k_mean = atm_df['k_mean'].to_list()
                values_k_n = atm_df['k_n'].to_list()

                np.put(stat_k_mean[day_of_week], idx, values_k_mean)
                np.put(stat_k_n[day_of_week], idx, values_k_n)

        output_dict[atm_id].update({'stat_k_mean': stat_k_mean,
                                    'stat_k_n': stat_k_n,
                                    'stat_sums': stat_sums[atm_id]})

    df.drop(columns=['day_of_week'], inplace=True)

    return output_dict


def calc_stats_offer(df: pd.DataFrame, TRAIN_START: datetime) -> defaultdict:
    """
    Function returns dict_1 (key=atm_id, value=matrix_of_prop for each denom (24x7))

    Input:
        - data: pd.DataFrame - filtered train DataFrame to process
            - atm_id         - int
            - dateTime_oper  - DateTime format for each transaction
            - note_5
            - note_10
            - note_20
            - note_50
            - note_100
            - note_200
            - note_500
    Output:
        - {'atm_id': np.array() (shape=24x7)}
    """

    def preprocess_for_stats(df: pd.DataFrame) -> pd.DataFrame:
        def date_filtering(df: pd.DataFrame, train_start, train_end):
            df = df[
                (df["dateTime_oper"] >= train_start)
                & (df["dateTime_oper"] <= train_end)
            ]
            return df

        TRAIN_END = df["dateTime_oper"].max() - timedelta(days=28)

        df = date_filtering(df, TRAIN_START, TRAIN_END)

        df = df.groupby([df["atm_id"], df["dateTime_oper"].dt.hour, df['day_of_week']]).agg(
            dict.fromkeys(agg_columns, "sum")
        )
        df.reset_index(inplace=True)
        df.columns = columns + agg_columns
        return df

    df['day_of_week'] = df['dateTime_oper'].apply(lambda row: row.weekday())

    output_dict = defaultdict(lambda: defaultdict(np.array))
    columns = ["atm_id", "hour", 'day_of_week']
    agg_columns = [
        "note_5",
        "note_10",
        "note_20",
        "note_50",
        "note_100",
        "note_200",
        "note_500",
    ]

    df = preprocess_for_stats(df)

    for atm_id in tqdm(df['atm_id'].unique()):
        stat_banknotes = np.zeros((7, 24, 7))  # day, hour, nominal
        for day_of_week in range(7):
            atm_df = df[(df['atm_id'] == atm_id) & (df['day_of_week'] == day_of_week)]

            total_banknotes = sum(atm_df[agg_columns].sum(axis=1))
            stats_per_hour = atm_df[agg_columns].to_numpy() / total_banknotes

            idx = atm_df['hour'].to_numpy()
            stat_banknotes[day_of_week][idx] = stats_per_hour

        output_dict[atm_id] = stat_banknotes.copy()

    df.drop(columns=['day_of_week'], inplace=True)
    return output_dict
