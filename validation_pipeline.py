from logger import get_logger
import hydra
from hydra.core.config_store import ConfigStore
from entities import ValidationConfig
import pandas as pd
from validation import get_metrics
from utils import load_csv

logger = get_logger('validation')

cs = ConfigStore.instance()
cs.store(name="validation_config", node=ValidationConfig)

# todo: total redo this validation step


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_validation_pipeline(cfg: ValidationConfig):
    logger.info("Reading Data: Start")
    preds_n_oper = pd.read_csv(cfg.predictions.preds_n_oper_path)
    preds_sum = pd.read_csv(cfg.predictions.preds_sum_path)
    preds_n_bills = pd.read_csv(cfg.predictions.preds_n_bills_path)

    test_n_oper = load_csv(cfg.data.test_n_oper)
    test_sum = load_csv(cfg.data.test_sum)
    test_n_bills = load_csv(cfg.data.test_n_bills)
    logger.info("Reading Data: Done")

    logger.info("Calculating test metrics: Start")
    metrics_n_oper = get_metrics(test_n_oper, preds_n_oper)
    metrics_sum = get_metrics(test_sum, preds_sum)
    metrics_n_bills = get_metrics(test_n_bills, preds_n_bills)
    logger.info("Calculating test metrics: Done")

    logger.info("Saving metrics: Start")
    metrics_n_oper.to_excel(cfg.metrics_postprocess.n_oper_path)
    metrics_sum.to_excel(cfg.metrics_postprocess.sum_path)
    metrics_n_bills.to_excel(cfg.metrics_postprocess.n_bills_path)
    logger.info("Saving metrics: Done")


# if __name__ == "__main__":
#     run_validation_pipeline()
