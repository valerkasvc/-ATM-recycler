import pandas as pd
from data import basic_preprocess, calc_stats_demand, calc_stats_offer
from logger import get_logger
from utils import save_dict_to_file, save_csv
import hydra
from datetime import datetime
from hydra.core.config_store import ConfigStore
from entities import PreprocessConfig
from data import validate_data, remove_useless_atms

logger = get_logger('preprocess')

cs = ConfigStore.instance()
cs.store(name="preprocess_config", node=PreprocessConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_preprocess_pipeline(cfg: PreprocessConfig) -> None:
    logger.info("Gathering raw data: Start")
    df_raw = pd.read_csv(cfg.data.raw_data_path, sep=";")  # todo: use gather_data.py for gathering f1.csv from cash-cycle
    logger.info("Gathering raw data: Done")

    logger.info("Basic preprocessing: Start")
    df_offer, df_demand = basic_preprocess(df_raw)
    logger.info("Basic preprocessing: Done")

    logger.info(f"Number of ATMs before validation: {df_offer['atm_id'].nunique()}")
    logger.info("Data Validation: Start")
    normal_atm, broken_atm = validate_data(df_offer, df_demand, logger, cfg.prediction_date, cfg.statistics.req_period)
    df_offer, df_demand = remove_useless_atms(df_offer, df_demand, normal_atm)
    logger.info("Data Validation: Done")
    logger.info(f"Number of ATMs after validation: {df_offer['atm_id'].nunique()}")

    logger.info("Saving demand/offer data: Start")
    save_csv(df_offer, cfg.data.offer_path)
    save_csv(df_demand, cfg.data.demand_path)
    logger.info("Saving demand/offer data: Done")

    logger.info("Gathering statistics: Start")
    train_start_date = datetime.strptime(cfg.train_start_date, "%d-%m-%Y")
    stats_demand = calc_stats_demand(df_demand, train_start_date)
    stats_offer = calc_stats_offer(df_offer, train_start_date)
    logger.info("Gathering statistics: Done")

    logger.info("Saving statistics: Start")
    save_dict_to_file(stats_offer.copy(), cfg.statistics.offer_path)
    save_dict_to_file(stats_demand.copy(), cfg.statistics.demand_path)
    logger.info("Saving statistics: Done")


if __name__ == "__main__":
    run_preprocess_pipeline()
