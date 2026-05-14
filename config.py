from dataclasses import dataclass


@dataclass(frozen=True)
class Data:
    raw_data_path: str
    offer_path: str
    demand_path: str
    train_n_oper: str
    test_n_oper: str
    train_sum: str
    test_sum: str
    train_n_bills: str
    test_n_bills: str


@dataclass(frozen=True)
class Statistics:
    offer_path: str
    demand_path: str
    demand_postprocess_path: str
    req_period: int


@dataclass(frozen=True)
class Predictions:
    preds_n_oper_path: str
    preds_sum_path: str
    preds_n_bills_path: str


@dataclass(frozen=True)
class Model:
    n_oper_path: str
    sum_path: str
    n_bills_path: str


@dataclass(frozen=True)
class Metrics:
    n_oper_path: str
    sum_path: str
    n_bills_path: str


@dataclass(frozen=True)
class MetricsPostprocess:
    n_oper_path: str
    sum_path: str
    n_bills_path: str


@dataclass(frozen=True)
class Report:
    n_oper_path: str
    sum_path: str
    n_bills_path: str


@dataclass(frozen=True)
class PreprocessConfig:
    data: Data
    statistics: Statistics
    prediction_date: str
    train_start_date: str


@dataclass(frozen=True)
class FitPredictConfig:
    data: Data
    predictions: Predictions
    statistics: Statistics
    model: Model
    metrics: Metrics
    report: Report
    horizon: int
    fit_models: bool
    prediction_date: str


@dataclass(frozen=True)
class PostProcessConfig:
    statistics: Statistics
    predictions: Predictions
    model: Model
    demand_path: str
    offer_path: str
    calc_demand_postpropcess_stats: bool
    max_total_per_hour: int
    max_count_of_operations_per_hour: int


@dataclass(frozen=True)
class ValidationConfig:
    data: Data
    predictions: Predictions
    metrics: Metrics
    metrics_postprocess: MetricsPostprocess
