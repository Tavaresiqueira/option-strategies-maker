# Python built-in libraries
from datetime import datetime
from typing import Dict

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

# Module-specific imports
from general_functions import (
    black_scholes_call,
    black_scholes_put,
    calculate_call_greeks,
    calculate_put_greeks,
    aggregate_greeks,
)

class StrategiesMaker:
    """
    This class provides methods to calculate and plot the Greeks for various option strategies.
    """

    def __init__(self, ticker: str, expiration_date: datetime, risk_free_rate: float, implied_volatility: float):
        """
        Initialize the StrategiesMaker class with stock ticker, expiration date, risk-free rate, and implied volatility.

        :param ticker: Stock ticker symbol.
        :param expiration_date: Expiration date of the options.
        :param risk_free_rate: Risk-free interest rate.
        :param implied_volatility: Implied volatility of the stock.
        """
        self.ticker = ticker
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.implied_volatility = implied_volatility
        self.stock_data = self._fetch_stock_data(ticker)

    def _fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical stock data for the given ticker.

        :param ticker: Stock ticker symbol.
        :return: DataFrame with historical stock data.
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        return data

    def calculate_butterfly_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate Greeks for a butterfly spread at a given stock price.

        :param S: Stock price.
        :param T: Time to expiration in years. If not provided, it defaults to the time until the expiration date.
        :return: A dictionary of Greek values.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        lower_strike = S * 0.95
        middle_strike = S
        upper_strike = S * 1.05

        greeks_lower = calculate_call_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_middle = calculate_call_greeks(S, middle_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_upper = calculate_call_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        return aggregate_greeks(greeks_lower, {k: -2*v for k, v in greeks_middle.items()}, greeks_upper)

    def calculate_iron_condor_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the total Greeks for an iron condor spread.

        :return: A dictionary of aggregated Greek values.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            current_price = self.stock_data['Close'].iloc[-1]

        # Defining strikes for iron condor spread
        lower_put_strike = current_price * 0.85
        upper_put_strike = current_price * 0.95
        lower_call_strike = current_price * 1.05
        upper_call_strike = current_price * 1.15

        # Calculating Greeks for each leg of the iron condor spread
        greeks_lower_put = calculate_put_greeks(current_price, lower_put_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_upper_put = calculate_put_greeks(current_price, upper_put_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_lower_call = calculate_call_greeks(current_price, lower_call_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_upper_call = calculate_call_greeks(current_price, upper_call_strike, T, self.risk_free_rate, self.implied_volatility)

        # Aggregating Greeks
        return aggregate_greeks(greeks_lower_put, greeks_upper_put, {k: -v for k, v in greeks_lower_call.items()}, {k: -v for k, v in greeks_upper_call.items()})

    def calculate_bull_put_spread_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks of a bull put spread options strategy.

        Parameters:
            S (float): The current price of the underlying asset.
            T (float): The time to expiration of the options, in years.

        Returns:
            Dict[str, float]: A dictionary containing the calculated Greeks of the bull put spread strategy.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        lower_strike = S * 0.95
        upper_strike = S * 1.05

        greeks_put_long = calculate_put_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put_short = calculate_put_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks(greeks_put_long, {k: -v for k, v in greeks_put_short.items()})

    def calculate_bear_put_spread_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a bear put spread options strategy.
        
        This function takes the current stock price (S) and time to expiration (T) as input parameters.
        It calculates the lower strike (lower_strike) as 95% of the current stock price and the upper strike (upper_strike) as 105% of the current stock price.
        
        It then calls the 'calculate_put_greeks' function twice, once for the upper strike and once for the lower strike, to calculate the Greeks (greeks_put_long and greeks_put_short).
        The 'calculate_put_greeks' function takes the current stock price, strike price, time to expiration, risk-free rate, and implied volatility as input parameters and returns a dictionary of Greeks.
        
        Finally, the function calls the 'aggregate_greeks' function to aggregate the Greeks of the long put option (greeks_put_long) and the short put option (greeks_put_short) and returns the resulting dictionary.
        
        Parameters:
        - S (float): The current stock price.
        - T (float): The time to expiration.
        
        Returns:
        - Dict[str, float]: A dictionary of Greeks for the bear put spread options strategy.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        lower_strike = S * 0.95
        upper_strike = S * 1.05

        greeks_put_long = calculate_put_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put_short = calculate_put_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks(greeks_put_long, {k: -v for k, v in greeks_put_short.items()})

    def calculate_bear_call_spread_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculates the Greeks for a bear call spread options strategy.

        Args:
            S (float): The current price of the underlying asset.
            T (float): The time to expiration of the options contracts.

        Returns:
            Dict[str, float]: A dictionary containing the aggregate Greeks for the bear call spread strategy.
        """

        if T is None:    
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        lower_strike = S * 0.95
        upper_strike = S * 1.05

        greeks_call_long = calculate_call_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_call_short = calculate_call_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_call_long.items()}, greeks_call_short)

    def calculate_long_straddle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a long straddle options strategy.

        Parameters:
            S (float): The spot price of the underlying asset.
            T (float): The time to expiration of the options.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated Greeks for the long straddle.

        """

        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        strike = S

        greeks_call = calculate_call_greeks(S, strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put = calculate_put_greeks(S, strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks(greeks_call, greeks_put)

    def calculate_long_strangle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a long strangle option strategy.

        Parameters:
            S (float): The current price of the underlying asset.
            T (float): The time to expiration of the options.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated Greeks for the long strangle option strategy.
        """

        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        lower_strike = S * 0.95
        upper_strike = S * 1.05

        greeks_call = calculate_call_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put = calculate_put_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks(greeks_call, greeks_put)

    def calculate_protective_collar_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a protective collar strategy.

        Args:
            S (float): The current price of the underlying asset.
            T (float): The time to expiration of the options.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated Greeks for the strategy.
        """

        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        call_strike = S * 1.05
        put_strike = S * 0.95

        greeks_call_short = calculate_call_greeks(S, call_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put_long = calculate_put_greeks(S, put_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_call_short.items()}, greeks_put_long)

    def plot_greeks_across_price_range(self, strategy: str, price_deviation: float = 0.2):
        """
        Plot Greeks across a range of stock prices for a specified strategy.

        :param strategy: The strategy to plot (e.g., 'butterfly', 'iron_condor', 'bull_call', etc.).
        :param price_deviation: Percentage deviation from the current stock price for the price range.
        """
        
        # Calculate lower and upper bounds of the price range
        current_price = self.stock_data['Close'].iloc[-1]
        lower_bound = current_price * (1 - price_deviation)
        upper_bound = current_price * (1 + price_deviation)
        
        # Generate a list of prices within the price range
        price_range = np.linspace(lower_bound, upper_bound, 100)
        
        # Calculate the time to expiration in years
        T = (self.expiration_date - datetime.now()).days / 365
        
        # Initialize lists to store greeks
        deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
        
        # Calculate greeks for each price in the price range
        for S in price_range:
            if strategy == 'butterfly':
                greeks = self.calculate_butterfly_greeks(S, T)
            elif strategy == 'iron_condor':
                greeks = self.calculate_iron_condor_greeks(S, T)
            elif strategy == 'bull_call':
                greeks = self.calculate_bull_call_spread_greeks(S, T)
            elif strategy == 'bull_put':
                greeks = self.calculate_bull_put_spread_greeks(S, T)
            elif strategy == 'bear_put':
                greeks = self.calculate_bear_put_spread_greeks(S, T)
            elif strategy == 'bear_call':
                greeks = self.calculate_bear_call_spread_greeks(S, T)
            elif strategy == 'long_straddle':
                greeks = self.calculate_long_straddle_greeks(S, T)
            elif strategy == 'long_strangle':
                greeks = self.calculate_long_strangle_greeks(S, T)
            elif strategy == 'protective_collar':
                greeks = self.calculate_protective_collar_greeks(S, T)
            else:
                raise ValueError("Invalid strategy. Choose among defined strategies.")
            
            # Append greek values to the corresponding lists
            deltas.append(greeks['Delta'])
            gammas.append(greeks['Gamma'])
            thetas.append(greeks['Theta'])
            vegas.append(greeks['Vega'])
            rhos.append(greeks['Rho'])
        
        # Plot the greeks
        self._plot_greeks(price_range, deltas, gammas, thetas, vegas, rhos)

    @staticmethod
    def _plot_greeks(price_range: np.array, deltas: list, gammas: list, thetas: list, vegas: list, rhos: list):
        """
        Plot the Greeks for a range of stock prices.

        :param price_range: Array of stock prices.
        :param deltas: Delta values.
        :param gammas: Gamma values.
        :param thetas: Theta values.
        :param vegas: Vega values.
        :param rhos: Rho values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(price_range, deltas, label='Delta')
        plt.plot(price_range, gammas, label='Gamma')
        plt.plot(price_range, thetas, label='Theta')
        plt.plot(price_range, vegas, label='Vega')
        plt.plot(price_range, rhos, label='Rho')
        plt.title('Greeks Across Different Prices')
        plt.xlabel('Stock Price')
        plt.ylabel('Greek Values')
        plt.legend()
        plt.show()