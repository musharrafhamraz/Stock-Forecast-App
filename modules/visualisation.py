import matplotlib.pyplot as plt

def plot_forecast(forecast):
    """
    Plot the forecasted values.
    :param forecast: Forecasted values from the model
    """
    if forecast is None:
        print("No forecast data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(forecast, label="Forecast", color="blue")
    plt.title("Stock Price Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()