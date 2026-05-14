import numpy as np
import pandas as pd


def devide_counts(total_count: float, stats_array: np.ndarray) -> np.ndarray:
    answer = stats_array * total_count
    answers_round = np.round(answer).astype(np.int64)

    n, m = answers_round.shape

    delta = answer - answers_round
    new_ar = delta.reshape(-1)

    indexes = np.arange(new_ar.shape[0])

    arr_with_idx = [val for val in zip(new_ar, indexes)]
    arr_with_idx.sort()

    total_count = round(total_count)
    delta_count = total_count - answers_round.sum()

    if delta_count < 0:
        for val, idx in arr_with_idx[:-delta_count]:
            answers_round[idx // n][idx % m] -= 1
    if delta_count > 0:
        for val, idx in arr_with_idx[-delta_count:]:
            answers_round[idx // n][idx % m] += 1
    return answers_round


def apply_offer(row, stats_offer: dict) -> np.ndarray:
    atm_id, target, day_of_week = int(row['atm_id']), row['target'], row['day_of_week']
    answers_round = devide_counts(target, stats_offer[atm_id][day_of_week])
    return answers_round


def run_offer_postprocess(df: pd.DataFrame, stats_offer: dict) -> pd.DataFrame:
    preds = df.copy()
    preds['dateTime_oper'] = pd.to_datetime(preds['dateTime_oper'])
    preds['day_of_week'] = preds['dateTime_oper'].apply(lambda row: row.weekday())

    notes = [5, 10, 20, 50, 100, 200, 500]
    n = preds.shape[0]

    preds["bins"] = preds.apply(lambda row: apply_offer(row, stats_offer), axis=1)
    preds = preds.explode("bins")
    preds["hour"] = list(range(24)) * n

    for idx, note in enumerate(notes):
        preds[f"note_{note}"] = preds["bins"].apply(lambda row: row[idx])

    preds.drop(["target", "bins", 'day_of_week'], axis=1, inplace=True)
    return preds
