from utils import load_dict_from_file, save_csv
import pandas as pd
from postprocess import (
    run_offer_postprocess,
    get_demand_postprocess_stats,
    run_demand_postprocess,
)
from logger import get_logger
import hydra
from hydra.core.config_store import ConfigStore
from entities import PostProcessConfig


logger = get_logger('postprocess')

cs = ConfigStore.instance()
cs.store(name="postprocess_config", node=PostProcessConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_postprocess_pipeline(cfg: PostProcessConfig) -> None:
    def run_offer() -> None:
        stats_offer = load_dict_from_file(cfg.statistics.offer_path)
        offer_postprocessed = run_offer_postprocess(preds_n_bills, stats_offer)
        save_csv(offer_postprocessed, cfg.offer_path)

    def run_demand() -> None:
        stats_demand = load_dict_from_file(cfg.statistics.demand_path)
        if not cfg.calc_demand_postpropcess_stats:
            try:
                demand_postprocess_stats = load_dict_from_file(cfg.statistics.demand_postprocess_path)
            except Exception as error:
                logger.critical(f'load_dict_from_file error: {error}')
                logger.critical("There may be no presaved demand_postprocess_stats")
                logger.critical("Set parameter calc_demand_postprocess_stats: True")
        else:
            demand_postprocess_stats = get_demand_postprocess_stats(stats_demand,
                                                                    cfg.statistics.demand_postprocess_path,
                                                                    cfg.max_count_of_operations_per_hour,
                                                                    cfg.max_total_per_hour)

        demand_preds = pd.merge(preds_sum, preds_n_oper, on=["atm_id", "dateTime_oper"])
        demand_preds.rename(
            columns={"target_x": "sum", "target_y": "n_oper", "dateTime_oper": "date"},
            inplace=True,
        )
        demand_postprocessed = run_demand_postprocess(demand_preds, demand_postprocess_stats, stats_demand).dropna()
        save_csv(demand_postprocessed, cfg.demand_path)

    logger.info("Reading predictions: Start")
    preds_n_oper = pd.read_csv(cfg.predictions.preds_n_oper_path)
    preds_sum = pd.read_csv(cfg.predictions.preds_sum_path)
    preds_n_bills = pd.read_csv(cfg.predictions.preds_n_bills_path)
    logger.info("Reading predictions: Done")

    logger.info("Running offer: Start")
    run_offer()
    logger.info("Running offer: Done")

    logger.info("Running demand: Start")
    run_demand()
    logger.info("Running demand: Done")


if __name__ == "__main__":
    run_postprocess_pipeline()
