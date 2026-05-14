import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

dct_money_to_idx = {5: 0, 10: 1, 20: 2, 50: 3, 100: 4, 200: 5, 500: 6}


def find_all_digits_in_string(x: str) -> list:
    answer = []
    digit = 0
    x += " "
    for char in x:
        if char.isdigit():
            digit *= 10
            digit += int(char)
        else:
            if digit != 0:
                answer.append(digit)
            digit = 0
    return answer


def parse_money(x: str) -> list:
    answer = [0] * 7
    arr = str(x).split("^")

    for value in arr:
        digits = find_all_digits_in_string(value)
        if len(digits) == 2:
            answer[dct_money_to_idx[digits[0]]] = digits[1]
    return answer


def basic_preprocess(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    def time_preprocess(df: pd.DataFrame) -> pd.DataFrame:
        date_format = "%d.%m.%Y %H:%M:%S"

        df["dateTime_start"] = pd.to_datetime(
            df["dateTime_start"], format=date_format, errors="coerce"
        )
        df["dateTime_finish"] = pd.to_datetime(
            df["dateTime_finish"], format=date_format, errors="coerce"
        )
        df["dateTime_oper"] = pd.to_datetime(
            df["dateTime_oper"], format=date_format, errors="coerce"
        )

        df["operation_time"] = df["dateTime_finish"] - df["dateTime_start"]
        max_time = df["operation_time"].quantile(0.9995)

        idx_to_delete = df[df["operation_time"] > max_time].index
        df = df[~df.index.isin(idx_to_delete)]

        df.drop(
            ["dateTime_start", "dateTime_finish", "operation_time"],
            axis=1,
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def filtering(df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            (df["currency"] == "BYN")
            & (df["cash_oper"] == 1)
            & (df["dateTime_oper"] >= datetime(2021, 11, 1))
        ]
        # todo: rewrite parsing data, wrong parsing (5000, 1000, 1)
        df.drop(index=df[df["notes"].str.find("1000") != -1].index, inplace=True)
        df.drop(index=df[df["notes"].str.find("5000") != -1].index, inplace=True)
        return df

    def parse_notes_column(df: pd.DataFrame) -> pd.DataFrame:
        df_denom = []
        for note in df["notes"].values:
            df_denom.append(parse_money(note))
        df_notes = pd.DataFrame(df_denom, columns=agg_columns)
        df = pd.concat([df, df_notes], axis=1)
        df.drop("notes", axis=1, inplace=True)
        return df

    def offer_demand_split(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        df_offer = df[df["oper_name"] == "CIN"][columns + ["notes"]]
        df_offer.reset_index(drop=True, inplace=True)

        df_demand = df[df["oper_name"] == "DIS"][columns]
        df_demand.reset_index(drop=True, inplace=True)
        return df_offer, df_demand

    columns = ["atm_id", "dateTime_oper", "sum"]
    agg_columns = [
        "note_5",
        "note_10",
        "note_20",
        "note_50",
        "note_100",
        "note_200",
        "note_500",
    ]

    df = time_preprocess(df)
    df = filtering(df)
    df_offer, df_demand = offer_demand_split(df)
    df_offer = parse_notes_column(df_offer)
    return df_offer, df_demand


def offer_train_test_split(data: pd.DataFrame, horizon: int, prediction_date: str) -> [pd.DataFrame, pd.DataFrame]:
    def daily_grouping(df: pd.DataFrame) -> pd.DataFrame:
        df["n_bills"] = df[agg_columns].sum(axis=1)
        df.set_index("dateTime_oper", inplace=True)
        df = df.groupby("atm_id").resample("D").agg({"n_bills": "sum"})
        df = df.reset_index(level=(0, 1))
        return df

    df = data.copy()
    horizon = timedelta(days=horizon)

    agg_columns = [
        "note_5",
        "note_10",
        "note_20",
        "note_50",
        "note_100",
        "note_200",
        "note_500",
    ]
    df = daily_grouping(df)
    df.rename(columns={"n_bills": "target"}, inplace=True)
    train = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)

    for atm_id in tqdm(df["atm_id"].unique()):
        atm_df = df[df["atm_id"] == atm_id]

        if prediction_date != "":
            SPLIT_DATE = datetime.strptime(prediction_date, "%d-%m-%Y")
        else:
            max_date = atm_df["dateTime_oper"].max()
            SPLIT_DATE = max_date - horizon

        train_df = atm_df.loc[atm_df["dateTime_oper"] <= SPLIT_DATE]
        test_df = atm_df.loc[atm_df["dateTime_oper"] > SPLIT_DATE]

        atm_shape = test_df[test_df["atm_id"] == atm_id].shape[0]
        if atm_shape != 28:
            print(f"Wrong shape for {atm_id}", atm_shape)

        train = pd.concat([train, train_df], axis=0)
        test = pd.concat([test, test_df], axis=0)
    return train, test


def demand_train_test_split(data: pd.DataFrame, horizon: int, prediction_date: str):
    def daily_grouping(df: pd.DataFrame) -> pd.DataFrame:
        df.set_index("dateTime_oper", inplace=True)
        df["n_oper"] = 1
        df = df.groupby("atm_id").resample("D").agg({"sum": "sum", "n_oper": "sum"})
        df = df.reset_index(level=(0, 1))
        return df

    df = data.copy()
    horizon = timedelta(days=horizon)

    columns = ["atm_id", "dateTime_oper", "target"]
    df = daily_grouping(df)

    df_n_oper = df[columns[:-1] + ["n_oper"]]
    df_n_oper.rename(columns={"n_oper": "target"}, inplace=True)

    df_sum = df[columns[:-1] + ["sum"]]
    df_sum.rename(columns={"sum": "target"}, inplace=True)

    train_n_oper, test_n_oper, train_sum, test_sum = [pd.DataFrame(columns=columns) for _ in range(4)]

    for atm_id in tqdm(df["atm_id"].unique()):
        atm_df_n_oper = df_n_oper[df_n_oper["atm_id"] == atm_id]
        atm_df_sum = df_sum[df_sum["atm_id"] == atm_id]
        assert atm_df_sum["dateTime_oper"].max() == atm_df_n_oper["dateTime_oper"].max()

        if prediction_date != "":
            SPLIT_DATE = datetime.strptime(prediction_date, "%d-%m-%Y")
        else:
            max_date = atm_df_sum["dateTime_oper"].max()
            SPLIT_DATE = max_date - horizon

        train_df_n_oper = atm_df_n_oper.loc[atm_df_n_oper["dateTime_oper"] <= SPLIT_DATE]
        test_df_n_oper = atm_df_n_oper.loc[atm_df_n_oper["dateTime_oper"] > SPLIT_DATE]
        train_df_sum = atm_df_sum.loc[atm_df_sum["dateTime_oper"] <= SPLIT_DATE]
        test_df_sum = atm_df_sum.loc[atm_df_sum["dateTime_oper"] > SPLIT_DATE]

        atm_shape = train_df_sum[train_df_sum["atm_id"] == atm_id].shape[0]
        assert atm_shape != 28, f"Wrong shape for {atm_id}, atm_shape: {atm_shape}"

        train_n_oper = pd.concat([train_n_oper, train_df_n_oper], axis=0)
        test_n_oper = pd.concat([test_n_oper, test_df_n_oper], axis=0)

        train_sum = pd.concat([train_sum, train_df_sum], axis=0)
        test_sum = pd.concat([test_sum, test_df_sum], axis=0)
    return train_n_oper, test_n_oper, train_sum, test_sum
