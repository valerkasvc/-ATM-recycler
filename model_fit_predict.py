import os
import abc
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from logging import Logger
import holidays
from datetime import datetime
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot


class AbstractModel(abc.ABC):
    def __init__(self, horizon: int):
        self.horizon = horizon

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X=None, y=None):
        raise NotImplementedError


class ProphetModel(AbstractModel):
    def __init__(self, horizon: int, metrics_path: str, logger: Logger, stats_req_period: int,
                 models_path: str = "models/", report_path: str = "reports/", outliers_std: float = 2.5):
        super().__init__(horizon)
        self.holid = self.get_holidays()
        self.outliers_std = outliers_std
        self.models_path = models_path
        self.metrics_path = metrics_path
        self.logger = logger
        self.report_path = report_path
        self.stats_req_period = stats_req_period

        params_grid = {'seasonality_mode': ['additive', 'multiplicative'],
                       'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3,  0.5],
                       'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                       'holidays_prior_scale': [0.01, 0.05, 0.1, 0.25, 0.5]}
        self.all_params = [dict(zip(params_grid.keys(), v)) for v in itertools.product(*params_grid.values())]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_y = pd.concat([X, y], axis=1)
        X_y.rename(columns={"dateTime_oper": "ds", "target": "y"}, inplace=True)

        metrics = pd.DataFrame(columns=['atm_id', 'mae', 'rmse', 'mape'])
        for atm_id in tqdm(X_y["atm_id"].unique()):
            X_y_atm = X_y[X_y["atm_id"] == atm_id][["ds", "y"]]
            model, best_params, best_metrics = self._fit_atm(X_y_atm, atm_id)

            self.logger.debug(f"Model {atm_id} parameters: {best_params}")
            self.save_model(model, atm_id)
            metrics = pd.concat([metrics, best_metrics])

        self.save_metrics(metrics)

    def _fit_atm(self, X_y, atm_id: int):
        X_y = self._fit_preprocess(X_y)
        mae_metrics = []
        total_metrics = []
        for params in self.all_params:
            m = Prophet(
                **params,
                weekly_seasonality=True,
                yearly_seasonality=True,
                holidays=self.holid,
            ).fit(X_y)
            df_cv = cross_validation(m, initial=f'{self.stats_req_period} days', horizon=f'{self.horizon} days',
                                     period=f'{int(self.horizon * 1.6)} days', parallel='processes')
            df_p = performance_metrics(df_cv, rolling_window=1, metrics=['mae', 'rmse', 'mape'])
            mae_metrics.append(df_p['mae'].values[0])
            total_metrics.append(df_p)
        idx_best_metric: int = np.argmin(mae_metrics)
        best_params: dict = self.all_params[idx_best_metric]

        model = Prophet(
            **best_params,
            weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=self.holid
        ).fit(X_y)

        best_metrics: pd.DataFrame = total_metrics[idx_best_metric]
        best_metrics['atm_id'] = atm_id
        return model, best_params, best_metrics

    def _fit_preprocess(self, X_y: pd.DataFrame):
        mean, std = X_y["y"].astype(int).describe()[["mean", "std"]]
        outliers_idx = list(np.where(X_y["y"] > mean + self.outliers_std * std)[0]) + \
                       list(np.where(X_y["y"] < mean - self.outliers_std * std)[0])
        X_y["y"] = np.where(X_y.index.isin(outliers_idx), None, X_y["y"])
        return X_y

    def predict(self, X=None, y=None) -> pd.DataFrame:
        forecast = pd.DataFrame(columns=["atm_id", "dateTime_oper", "target"])
        for atm_id in tqdm(os.listdir(self.models_path)):
            if atm_id.startswith("."):
                continue
            model: Prophet = self.load_model(self.models_path + atm_id)
            if X is None:
                X = model.make_future_dataframe(self.horizon, include_history=False)
            forecast_atm = self._predict_atm(model, X, y)
            atm_id = int(atm_id[:-5])

            atm_id_df = pd.DataFrame([atm_id] * self.horizon, columns=["atm_id"])
            date = X.copy()
            date.rename(columns={"ds": "dateTime_oper"}, inplace=True)

            forecast_atm = pd.concat([atm_id_df, date, forecast_atm], axis=1)
            forecast = pd.concat([forecast, forecast_atm], axis=0)
            self.report_model_action(model, atm_id, forecast_atm)
        return forecast

    def _predict_atm(self, model: Prophet, X: pd.DataFrame, y=None) -> pd.DataFrame:
        forecast = model.predict(X)[["yhat"]].clip(0)
        forecast.rename(columns={"yhat": "target"}, inplace=True)
        return forecast

    def report_model_action(self, model: Prophet, atm_id: int, forecast_atm: pd.DataFrame) -> None:
        def report_model_components():
            components = model.plot_components(fcst)
            components.savefig(self.report_path + f'{atm_id}/' + 'components.png', dpi=150)

        def report_model_predictions():
            dates_min = model.history_dates[len(model.history_dates) // 1.4]
            fcst_cut = fcst[fcst['ds'] >= dates_min]

            fig, axs = plt.subplots(2, 1, figsize=(16, 10))
            model.plot(fcst, xlabel='Time', ylabel='y', ax=axs[0], include_legend=True)
            add_changepoints_to_plot(axs[0], model, fcst)
            axs[0].plot(forecast_atm['dateTime_oper'], forecast_atm['target'], c='darkgreen', label='Future forecast')
            axs[0].legend()

            model.plot(fcst_cut, xlabel='Time', ylabel='y', ax=axs[1], include_legend=True)
            add_changepoints_to_plot(axs[1], model, fcst)
            axs[1].plot(forecast_atm['dateTime_oper'], forecast_atm['target'], c='darkgreen', label='Future forecast')
            axs[1].set_xlim(dates_min)
            axs[1].legend()

            fig.savefig(self.report_path + f'{atm_id}/' + '/predictions.png', dpi=150)

        os.makedirs(self.report_path + f'{atm_id}/', exist_ok=True)
        fcst = model.predict()

        report_model_components()
        report_model_predictions()

    def save_model(self, model, atm_id: int) -> None:
        os.makedirs(self.models_path, exist_ok=True)
        with open(self.models_path + str(atm_id) + ".json", "w") as fout:
            fout.write(model_to_json(model))

    def save_metrics(self, metrics: pd.DataFrame) -> None:
        metrics.set_index('atm_id', inplace=True)
        metrics['mape'] = metrics['mape'] * 100
        total_stat = pd.DataFrame([metrics.mean().values], columns=metrics.columns, index=['Total'])
        metrics = pd.concat([metrics, total_stat], axis=0)
        metrics.to_excel(self.metrics_path)

    @staticmethod
    def load_model(path: str):
        with open(path, "r") as fin:
            model = model_from_json(fin.read())
        return model

    @staticmethod
    def get_holidays() -> pd.DataFrame:
        year_now = datetime.now().year
        holid = holidays.Belarus(years=[year_now - 1, year_now, year_now + 1])
        holid = pd.DataFrame.from_dict(holid, orient="index", columns=["holiday"])
        holid.reset_index(inplace=True)
        holid.rename(columns={"index": "ds"}, inplace=True)
        return holid
