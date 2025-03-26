from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from modules.data_handler import fetch_stock_data
from modules.model_handler import run_model
from modules.visualisation import plot_forecast

class StockForecastApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Input fields
        self.stock_input = TextInput(hint_text="Enter stock symbols (e.g., AAPL or JPM BAC)", multiline=False)
        self.layout.add_widget(self.stock_input)

        # Dropdown for model selection
        self.model_spinner = Spinner(text="Select Model", values=("ADF", "ARIMA", "LSTM", "SARIMA", "VECM"))
        self.layout.add_widget(self.model_spinner)

        # Dropdown for historical data selection
        self.history_spinner = Spinner(text="Select Years of Data", values=("5", "10", "15", "20"))
        self.layout.add_widget(self.history_spinner)

        # Dropdown for prediction horizon
        self.prediction_spinner = Spinner(text="Select Prediction Horizon", values=("1", "3", "6", "12"))
        self.layout.add_widget(self.prediction_spinner)

        # Run button
        self.run_button = Button(text="Run Forecast", size_hint=(1, 0.2))
        self.run_button.bind(on_press=self.run_forecast)
        self.layout.add_widget(self.run_button)

        # Output label
        self.output_label = Label(text="Results will appear here")
        self.layout.add_widget(self.output_label)

        return self.layout

    def run_forecast(self, instance):
        # Get user inputs
        stocks = self.stock_input.text.split()
        model = self.model_spinner.text
        years = int(self.history_spinner.text)
        horizon = int(self.prediction_spinner.text)

        # Validate inputs
        if model == "VECM" and len(stocks) != 2:
            self.output_label.text = "Error: VECM requires exactly two stock symbols."
            return

        if len(stocks) > 2:
            self.output_label.text = "Error: Please enter 1 or 2 stock symbols."
            return

        # Fetch stock data
        try:
            data = fetch_stock_data(stocks, years)
        except Exception as e:
            self.output_label.text = f"Error fetching stock data: {str(e)}"
            return

        # Run selected model
        try:
            forecast, mae, rmse = run_model(data, model, horizon)
        except Exception as e:
            self.output_label.text = f"Error running model: {str(e)}"
            return

        # Check if forecast is valid before plotting
        if forecast is not None:
            plot_forecast(forecast)
        else:
            self.output_label.text = "This model does not produce a forecast."

        # Update output label
        if mae is not None and rmse is not None:
            self.output_label.text += f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}"

if __name__ == "__main__":
    StockForecastApp().run()