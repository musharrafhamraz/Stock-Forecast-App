def calculate_metrics(actual, predicted):
    """
    Calculate MAE and RMSE.
    :param actual: Actual values
    :param predicted: Predicted values
    :return: MAE, RMSE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse