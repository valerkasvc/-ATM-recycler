import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_metrics(true: pd.DataFrame, pred: pd.DataFrame):
    def color_krpf(value):
        return 'color:red' if value < 80 else 'color:green'

    def color_r2(value):
        return 'color:red' if value < 0.25 else 'color:green'

    def _smape(true, pred):
        try:
            return (100 / true.shape[0]) * sum((pred - true) * 2 / (abs(pred) + abs(true) + eps))
        except:
            return None

    def _krpf(true, pred):
        try:
            return (100 / true.shape[0]) * sum(1 - (abs(true - pred)) / (true + pred + eps))
        except:
            return None

    def _mbe(true, pred):
        return sum(pred - true)

    metrics = pd.DataFrame()
    eps = 10e-8

    for atm_id in list(true['atm_id'].unique()):
        true_atm = true[true['atm_id'] == atm_id]
        pred_atm = pred[pred['atm_id'] == atm_id]
        if true_atm.shape[0] != pred_atm.shape[0]:
            print(
                f'Wrong values of preds, atm_id: {atm_id}, pred.shape: {pred_atm.shape[0]}, true.shape: {true_atm.shape[0]}')
            continue
        vals = true_atm['target'].values, pred_atm['target'].values
        r2 = r2_score(*vals)
        mae = mean_absolute_error(*vals)
        mse = mean_squared_error(*vals)
        mbe = _mbe(*vals)
        smape = _smape(*vals)
        krpf = _krpf(*vals)

        atm_metrics = pd.DataFrame({'mbe': [mbe],
                                    'mae': [mae],
                                    'rmse': [np.sqrt(mse)],
                                    'mse': [mse],
                                    'R2': [r2],
                                    'smape': [smape],
                                    'krpf': [krpf]},
                                   index=[f'{atm_id}'])
        metrics = pd.concat([metrics, atm_metrics], axis=0)

    total_stat = pd.DataFrame([metrics.mean().values], columns=metrics.columns, index=['Total'])
    metrics = pd.concat([metrics, total_stat], axis=0)
    return (metrics
            .sort_index()
            .style
            .applymap(color_krpf, subset=['krpf'])
            .applymap(color_r2, subset=['R2']))
