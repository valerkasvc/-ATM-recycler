import pandas as pd
from data import demand_train_test_split, offer_train_test_split
from utils import save_csv
from models import ProphetModel
from logger import get_logger
import logging
import hydra
from hydra.core.config_store import ConfigStore
from entities import FitPredictConfig
import time


logger = get_logger('fit_predict')

cs = ConfigStore.instance()
cs.store(name="fit_predict_config", node=FitPredictConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_fit_predict_pipeline(cfg: FitPredictConfig) -> None:
    def basic_predictions_test() -> None:
        for atm_id in train_n_oper["atm_id"].unique():
            assert train_n_oper[train_n_oper["atm_id"] == atm_id]["dateTime_oper"].max() == \
                   train_sum[train_sum["atm_id"] == atm_id]["dateTime_oper"].max()

        for atm_id in preds_n_oper["atm_id"].unique():
            assert preds_n_oper[preds_n_oper["atm_id"] == atm_id].shape[0] == \
                   preds_sum[preds_sum["atm_id"] == atm_id].shape[0]
            assert preds_n_oper[preds_n_oper["atm_id"] == atm_id]["dateTime_oper"].max() == \
                   preds_sum[preds_sum["atm_id"] == atm_id]["dateTime_oper"].max()
            assert preds_n_oper[preds_n_oper["atm_id"] == atm_id]["dateTime_oper"].min() == \
                   preds_sum[preds_sum["atm_id"] == atm_id]["dateTime_oper"].min()

            assert preds_n_bills[preds_n_bills["atm_id"] == atm_id].shape[0] == \
                   preds_sum[preds_sum["atm_id"] == atm_id].shape[0]
            assert preds_n_bills[preds_n_bills["atm_id"] == atm_id].shape[0] == \
                   preds_n_oper[preds_n_oper["atm_id"] == atm_id].shape[0]

    def run_init_models():
        model_n_oper = ProphetModel(horizon=cfg.horizon, logger=logger, stats_req_period=cfg.statistics.req_period,
                                    metrics_path=cfg.metrics.n_oper_path,
                                    models_path=cfg.model.n_oper_path, report_path=cfg.report.n_oper_path)
        model_sum = ProphetModel(horizon=cfg.horizon, logger=logger, stats_req_period=cfg.statistics.req_period,
                                 metrics_path=cfg.metrics.sum_path,
                                 models_path=cfg.model.sum_path, report_path=cfg.report.sum_path,)
        model_n_bills = ProphetModel(horizon=cfg.horizon, logger=logger, stats_req_period=cfg.statistics.req_period,
                                     metrics_path=cfg.metrics.n_bills_path,
                                     models_path=cfg.model.n_bills_path, report_path=cfg.report.n_bills_path,)
        return model_n_oper, model_sum, model_n_bills

    def run_fit(model_n_oper: ProphetModel, model_sum: ProphetModel, model_n_bills: ProphetModel):
        model_n_oper.fit(X=train_n_oper[["atm_id", "dateTime_oper"]], y=train_n_oper["target"])
        model_sum.fit(X=train_sum[["atm_id", "dateTime_oper"]], y=train_sum["target"])
        model_n_bills.fit(X=train_n_bills[["atm_id", "dateTime_oper"]], y=train_n_bills["target"])
        return model_n_oper, model_sum, model_n_bills

    def run_predict(model_n_oper: ProphetModel, model_sum: ProphetModel, model_n_bills: ProphetModel):
        preds_n_oper = model_n_oper.predict()
        preds_sum = model_sum.predict()
        preds_n_bills = model_n_bills.predict()
        return preds_n_oper, preds_sum, preds_n_bills

    logging.getLogger("prophet").disabled = True
    logging.getLogger("cmdstanpy").disabled = True

    logger.info("Reading demand/offer data: Start")
    df_demand = pd.read_csv(cfg.data.demand_path, parse_dates=["dateTime_oper"])
    df_offer = pd.read_csv(cfg.data.offer_path, parse_dates=["dateTime_oper"])
    logger.info("Reading demand/offer data: Done")

    logger.info("Train/test splitting: Start")
    train_n_oper, test_n_oper, train_sum, test_sum = demand_train_test_split(df_demand, cfg.horizon, cfg.prediction_date)
    train_n_bills, test_n_bills = offer_train_test_split(df_offer, cfg.horizon, cfg.prediction_date)
    logger.info("Train/test splitting: Done")

    start = time.time()
    logger.info("Running fit pipeline: Start")
    models = run_init_models()
    if not cfg.fit_models:
        logger.critical("Be sure that you have traine & saved models on write splitted samples")
    else:
        models = run_fit(*models)
    logger.info("Running fit pipeline: Done")
    logger.info(f'Running fit pipeline time: {time.time() - start}')

    logger.info("Running predict pipeline: Start")
    preds_n_oper, preds_sum, preds_n_bills = run_predict(*models)
    logger.info("Running predict pipeline: Done")

    logger.info("Testing predection: Start")
    basic_predictions_test()
    logger.info("Testing predection: Done")

    logger.info("Saving prediction: Start")
    save_csv(preds_n_oper, cfg.predictions.preds_n_oper_path)
    save_csv(preds_sum, cfg.predictions.preds_sum_path)
    save_csv(preds_n_bills, cfg.predictions.preds_n_bills_path)
    logger.info("Saving prediction: Done")

    logger.info("Saving train/test: Start")
    save_csv(train_n_oper, cfg.data.train_n_oper)
    save_csv(train_sum, cfg.data.train_sum)
    save_csv(train_n_bills, cfg.data.train_n_bills)

    save_csv(test_n_oper, cfg.data.test_n_oper)
    save_csv(test_sum, cfg.data.test_sum)
    save_csv(test_n_bills, cfg.data.test_n_bills)
    logger.info("Saving train/test: Done")


if __name__ == "__main__":
    run_fit_predict_pipeline()
