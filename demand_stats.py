import numpy as np
import pandas as pd
import itertools
from collections import Counter
from utils import save_dict_to_file
from tqdm import tqdm


ADDITIVE = 100  # будем домножать на это число, чтобы не уходить в очень маленькие числа
EPS = 10e-8  # добавка для нулевых вероятностей


def get_demand_postprocess_stats(stats_demand: pd.DataFrame,
                                 demand_postprocess_path: str,
                                 max_count_of_operations_per_hour: int,
                                 max_total_per_hour: int) -> dict:
    demand_postprocess_stats = {
        atm_id: run_demand_postprocess_stats(stats_demand[atm_id]["stat_sums"],
                                             max_count_of_operations_per_hour, max_total_per_hour)
        for atm_id in tqdm(list(stats_demand.keys()))
    }
    save_dict_to_file(demand_postprocess_stats, demand_postprocess_path)
    return demand_postprocess_stats


def run_demand_postprocess_stats(sum_probabilities: np.array,
                                 max_count_of_operations: int,
                                 max_total: int) -> np.ndarray:
    sum_probabilities_copy = sum_probabilities.copy()
    sum_probabilities_copy[1:] += EPS

    shape = (max_count_of_operations + 1, max_total // 5 + 1)

    n, m = shape

    matrix_probabilities = np.zeros(shape)
    matrix_last_sums = np.zeros(shape)

    sum_probabilities_copy.resize(m)

    matrix_probabilities[1] = sum_probabilities_copy
    matrix_last_sums[1] = np.arange(0, m * 5, 5)

    for number_of_operations in range(2, n):
        for total_sum_idx in range(1, m):
            max_probability = 0
            sum_for_max_probability = 0

            for previous_total_idx in range(1, total_sum_idx):
                max_probability_for_last_n = matrix_probabilities[
                    number_of_operations - 1
                ][previous_total_idx]
                probability_for_one = matrix_probabilities[1][
                    total_sum_idx - previous_total_idx
                ]

                if (
                    max_probability_for_last_n * probability_for_one * ADDITIVE
                    > max_probability
                ):
                    max_probability = (
                            max_probability_for_last_n * probability_for_one * ADDITIVE
                    )
                    sum_for_max_probability = (total_sum_idx - previous_total_idx) * 5

            matrix_probabilities[number_of_operations][total_sum_idx] = max_probability
            matrix_last_sums[number_of_operations][
                total_sum_idx
            ] = sum_for_max_probability

    return matrix_last_sums


def total_to_sums(total: int, number_of_operations: int, matrix: np.ndarray) -> list:
    answer = []

    if number_of_operations == 0 and total > 0:
        number_of_operations += 1

    total = min(total, (matrix.shape[1] - 2) * 5)
    number_of_operations = min(number_of_operations, matrix.shape[0] - 1)

    for idx in range(number_of_operations, 0, -1):
        sum = int(matrix[idx][total // 5])
        total -= sum
        answer.append(sum)

    return Counter(answer)


def devide_sums(total: float, stats_array: np.array) -> np.array:
    answer = stats_array * total
    answer /= 5
    answer = np.round(answer) * 5
    return answer.astype(np.int64)


def devide_counts(total_count: float, stats_array: np.array) -> np.array:
    answer = stats_array * total_count
    answer = np.round(answer).astype(np.int64)
    return answer


def duplicate(array, times=24):
    for val in array:
        for _ in range(times):
            yield val


def apply_function(row, demand_postprocess_stats: dict, stats_demand: dict) -> pd.Series:
    total = row['sum']
    total_count = row['n_oper']
    atm_id = row['atm_id']
    day_of_week = row['day_of_week']

    matrix = demand_postprocess_stats[atm_id]

    sums = devide_sums(total, stats_demand[atm_id]['stat_k_mean'][day_of_week])
    counts = devide_counts(total_count, stats_demand[atm_id]['stat_k_n'][day_of_week])

    return pd.Series(
        [
            atm_id,
            row["date"],
            [total_to_sums(sum, count, matrix) for sum, count in zip(sums, counts)],
        ],
        index=["atm_id", "date", "dict_sums"],
    )


def unpack(row, idx: int):
    if isinstance(row, tuple):
        return row[idx]
    else:
        return None


def run_demand_postprocess(data_predictions: pd.DataFrame,
                           demand_postprocess_stats: dict,
                           stats_demand: dict) -> pd.DataFrame:

    preds = data_predictions.copy()
    preds['date'] = pd.to_datetime(preds['date'])
    preds['day_of_week'] = preds['date'].apply(lambda row: row.weekday())

    answers = preds.apply(
        lambda row: apply_function(row, demand_postprocess_stats, stats_demand), axis=1
    )

    data = {
        "atm_id": duplicate(answers["atm_id"]),
        "date": duplicate(answers["date"]),
        "hour": list(range(24)) * answers.shape[0],
        "sums_dict": (
            [(key, value) for (key, value) in dct.items()]
            for dct in itertools.chain.from_iterable(answers["dict_sums"])
        ),
    }

    answers = pd.DataFrame(data)
    answers = answers.explode("sums_dict")
    answers["sum"] = answers["sums_dict"].apply(lambda row: unpack(row, 0))
    answers["count"] = answers["sums_dict"].apply(lambda row: unpack(row, 1))

    answers.drop(["sums_dict"], axis=1, inplace=True)

    return answers
