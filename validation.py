import pandas as pd
from logging import Logger
from datetime import timedelta
from typing import Tuple


def remove_useless_atms(df_offer: pd.DataFrame, df_demand: pd.DataFrame, normal_atm: set) -> [pd.DataFrame, pd.DataFrame]:
    offer_mask = df_offer['atm_id'].isin(normal_atm)
    demand_mask = df_demand['atm_id'].isin(normal_atm)
    df_offer = df_offer[offer_mask]
    df_demand = df_demand[demand_mask]
    return df_offer, df_demand


def get_dates(df: pd.DataFrame) -> list:
    for atm_id in df['atm_id'].unique():
        max_date = df[df['atm_id'] == atm_id]['dateTime_oper'].max()
        min_date = df[df['atm_id'] == atm_id]['dateTime_oper'].min()
        yield atm_id, min_date, max_date


def validate_data(df_offer: pd.DataFrame, df_demand: pd.DataFrame, logger: Logger,
                  prediction_date: str, req_period: int) -> Tuple[set, set]:
    def test_data_synchronicity() -> set:
        offer_unique = set(df_offer['atm_id'].unique())
        demand_unique = set(df_demand['atm_id'].unique())
        broken_atm = set(total_atm - offer_unique).union(total_atm - demand_unique)

        if len(broken_atm) != 0:
            logger.warning('ATMs which is not presented in all demand and offer datasets')
            logger.warning('There will be no predictions for them')
            logger.warning(broken_atm)
        return total_atm - broken_atm

    def test_data_history() -> set:
        broken_atm = set()

        for atm_id, min_date, max_date in get_dates(df_offer):
            if max_date - min_date < timedelta(days=req_period):
                broken_atm.add(atm_id)

        for atm_id, min_date, max_date in get_dates(df_demand):
            if max_date - min_date < timedelta(days=req_period):
                broken_atm.add(atm_id)

        if len(broken_atm) != 0:
            logger.warning(f'ATMs which have history less than {req_period} days will be removed')
            logger.warning('There will be no predictions for them')
            logger.warning(broken_atm)
        return total_atm - broken_atm

    def test_data_last_date() -> set:
        broken_atm = set()

        for atm_id, _, max_date in get_dates(df_offer):
            if not pd.to_datetime(prediction_date) <= max_date + timedelta(days=1):
                broken_atm.add(atm_id)

        for atm_id, _, max_date in get_dates(df_demand):
            if not pd.to_datetime(prediction_date) <= max_date + timedelta(days=1):
                broken_atm.add(atm_id)

        if len(broken_atm) != 0:
            logger.warning('ATMs which have max date less then prediction_date will be removed')
            logger.warning('There will be no predictions for them')
            logger.warning(broken_atm)
        return total_atm - broken_atm

    offer_unique = set(df_offer['atm_id'].unique())
    demand_unique = set(df_demand['atm_id'].unique())
    total_atm = offer_unique.union(demand_unique)

    atm_passed_history: set = test_data_history()
    atm_passed_synchronicity: set = test_data_synchronicity()

    normal_atm = atm_passed_history.intersection(atm_passed_synchronicity)

    if prediction_date != "":
        atm_passed_prediction_date: set = test_data_last_date()
        normal_atm = normal_atm.intersection(atm_passed_prediction_date)

    broken_atm = total_atm - normal_atm
    logger.info('ATMs which have passed all tests.')
    logger.info(normal_atm)
    if len(normal_atm) == 0:
        logger.critical("There is no ATms data which passed validation.")
    return normal_atm, broken_atm
