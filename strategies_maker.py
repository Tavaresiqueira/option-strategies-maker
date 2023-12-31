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
    calculate_call_greeks,
    calculate_put_greeks,
    aggregate_greeks
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

        The butterfly spread is a strategy that involves buying one call option at a lower strike price,
        selling two call options at the middle strike price, and buying one call option at a higher strike price.
        It is used when the trader expects the underlying asset to remain relatively stable.

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

        This function calculates the total Greeks for an iron condor spread strategy.
        The iron condor spread involves selling an out-of-the-money put and an out-of-the-money call option,
        while simultaneously buying an even further out-of-the-money put and call option.
        The strategy profits from the underlying asset staying within a certain price range.

        :param S: The current price of the underlying asset.
        :param T: The time to expiration of the options.
                If not provided, it is calculated as the difference between the expiration date and the current date.
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
        return aggregate_greeks(greeks_lower_put, {k: -v for k, v in greeks_upper_put.items()}, {k: -v for k, v in greeks_lower_call.items()}, greeks_upper_call)

    def calculate_bull_put_spread_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks of a bull put spread options strategy.

        This function calculates the Greeks of a bull put spread options strategy.
        The bull put spread involves selling a put option with a higher strike price
        and buying a put option with a lower strike price.

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

        The bear put spread strategy involves buying a put option with a lower strike price and selling a put option with a higher strike price.
        The goal of this strategy is to profit from a decrease in the price of the underlying stock, while limiting potential losses.

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

        This function calculates the Greeks, such as Delta, Gamma, Theta, and Vega, for a bear call spread options strategy.
        The bear call spread strategy involves selling a call option with a lower strike price and buying a call option 
        with a higher strike price, both with the same expiration date. The goal of this strategy is to profit from a 
        decrease in the price of the underlying asset.

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

        lower_strike = S * 1.05
        upper_strike = S * 1.15

        greeks_call_long = calculate_call_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_call_short = calculate_call_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_call_short.items()}, greeks_call_long)

    def calculate_long_straddle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a long straddle options strategy.

        The long straddle strategy involves buying both a call option and a put option with the same strike price and expiration date.
        This strategy profits from significant movements in the underlying asset's price, regardless of direction.
        The goal is to benefit from increased volatility.

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
    
    def calculate_short_straddle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a short straddle options strategy.

        A short straddle is an options strategy where a trader sells both a call option and a put option with the same
        strike price and expiration date. The strategy profits from a lack of significant movement in the underlying asset
        price.

        Parameters:
            S (float): The spot price of the underlying asset.
            T (float): The time to expiration of the options.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated Greeks for the short straddle.

        """

        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        strike = S

        greeks_call = calculate_call_greeks(S, strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put = calculate_put_greeks(S, strike, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_call.items()}, {k: -v for k, v in greeks_put.items()})

    def calculate_long_strangle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a long strangle option strategy.

        The long strangle strategy involves buying both a call option and a put option on the same underlying asset 
        with different strike prices. The goal is to profit from a significant price movement in either direction.

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

    def calculate_short_strangle_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a short strangle option strategy.

        The short strangle strategy involves selling both a call option and a put option on the same underlying asset 
        with different strike prices. The goal is to profit from a non-significant price movement.

        Parameters:
            S (float): The current price of the underlying asset.
            T (float): The time to expiration of the options.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated Greeks for the short strangle option strategy.
        """

        # Set default values for S and T if not provided
        T = T or (self.expiration_date - datetime.now()).days / 365
        S = S or self.stock_data['Close'].iloc[-1]

        # Calculate the lower and upper strike prices
        lower_strike = S * 0.95
        upper_strike = S * 1.05

        # Calculate the option Greeks for the call and put options
        greeks_call = calculate_call_greeks(S, upper_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_put = calculate_put_greeks(S, lower_strike, T, self.risk_free_rate, self.implied_volatility)

        # Aggregate the Greeks for the short strangle strategy
        return aggregate_greeks({k: -v for k, v in greeks_call.items()}, {k: -v for k, v in greeks_put.items()})
    
    def calculate_protective_collar_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate the Greeks for a protective collar strategy.

        The protective collar strategy involves buying a put option to protect against a decrease in the price of the underlying asset,
        while simultaneously selling a call option to offset the cost of the put option. This strategy limits both potential gains and losses.

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
    
    def calculate_fig_leaf_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate Greeks for a Fig Leaf (Leveraged Covered Call) strategy at a given stock price.

        :param S: Stock price.
        :param T: Time to expiration in years. If not provided, it defaults to the time until the expiration date.
        :return: A dictionary of Greek values.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define the strike price for the call option you sell
        sold_call_strike = S * 1.05  # Example: 5% above current price

        # Define the strike price for the longer-term in-the-money call option you buy
        bought_call_strike = S * 0.95  # Example: 5% below current price

        # Calculate Greeks for both the sold and bought call options
        greeks_sold_call = calculate_call_greeks(S, sold_call_strike, T, self.risk_free_rate, self.implied_volatility)
        greeks_bought_call = calculate_call_greeks(S, bought_call_strike, T, self.risk_free_rate, self.implied_volatility)
        
        return aggregate_greeks({k: -v for k, v in greeks_sold_call.items()}, greeks_bought_call)
    
    def calculate_backspread_with_calls_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate Greeks for a Backspread with Calls strategy.

        This strategy involves selling one call at a lower strike price and buying two calls at a higher strike price.

        Args:
            S (float): The stock price.
            T (float): The time to expiration in years.

        Returns:
            Dict[str, float]: A dictionary containing the calculated Greeks.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define the strike price for the call options
        strike_call_sold = S  # ATM option sold
        strike_call_bought = S * 1.1  # OTM option bought

        # Calculate the Greeks for the sold call option
        greeks_call_sold = calculate_call_greeks(S, strike_call_sold, T, self.risk_free_rate, self.implied_volatility)

        # Calculate the Greeks for the bought call option (assuming we buy twice as many)
        greeks_call_bought = calculate_call_greeks(S, strike_call_bought, T, self.risk_free_rate, self.implied_volatility)
        greeks_call_bought = {k: 2 * v for k, v in greeks_call_bought.items()}  # Double the Greeks for the bought option

        # Aggregate the Greeks
        return aggregate_greeks({k: -v for k, v in greeks_call_sold.items()}, greeks_call_bought)

    def calculate_frontspread_with_calls_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """

        This function calculates the Greeks for a frontspread options strategy using call options. The frontspread strategy involves buying an at-the-money (ATM) call option and selling two out-of-the-money (OTM) call option at a strike price 10% higher. The function takes two optional parameters:

        - `S` (float): The underlying stock price. If not provided, the function uses the last closing price from the `stock_data` attribute of the class instance.
        - `T` (float): The time to expiration in years. If not provided, the function calculates it based on the current date and the expiration date from the class instance.

        Returns:
        - A dictionary containing the aggregated Greeks for the frontspread strategy. The keys of the dictionary are the Greek names (delta, gamma, theta, vega), and the values are the corresponding values.

        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define the strike price for the call option
        strike_call_bought = S  # ATM option bought
        strike_call_sold = S * 1.1  # OTM option sold

        # Greeks for the bought call option
        greeks_call_bought = calculate_call_greeks(S, strike_call_bought, T, self.risk_free_rate, self.implied_volatility)

        # Greeks for the sold call option (assuming we buy twice as many)
        greeks_call_sold = calculate_call_greeks(S, strike_call_sold, T, self.risk_free_rate, self.implied_volatility)
        greeks_call_sold = {k: 2 * v for k, v in greeks_call_sold.items()}  # Double the Greeks for the bought option

        # Aggregate Greeks
        return aggregate_greeks({k: -v for k, v in greeks_call_sold.items()}, greeks_call_bought)

    def calculate_backspread_with_puts_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define strikes for the put options
        strike_put_sold = S  # ATM option sold
        strike_put_bought = S * 0.9  # ITM option bought

        # Greeks for the sold put option
        greeks_put_sold = calculate_put_greeks(S, strike_put_sold, T, self.risk_free_rate, self.implied_volatility)

        # Greeks for the bought put option
        greeks_put_bought = calculate_put_greeks(S, strike_put_bought, T, self.risk_free_rate, self.implied_volatility)
        greeks_put_bought = {k: 2 * v for k, v in greeks_put_bought.items()}  # Assume buying more puts than selling

        return aggregate_greeks({k: -v for k, v in greeks_put_sold.items()}, greeks_put_bought)
    
    def calculate_frontspread_with_puts_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculate Greeks for a Frontspread with Puts strategy.
        This strategy involves buying more put options than selling.
        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define the strike price for the put option
        strike_put_bought = S  # ATM option bought
        strike_put_sold = S * 0.9  # ITM option sold

        # Greeks for the bought put option
        greeks_put_bought = calculate_put_greeks(S, strike_put_bought, T, self.risk_free_rate, self.implied_volatility)

        # Greeks for the sold put option (assuming we buy twice as many)
        greeks_put_sold = calculate_put_greeks(S, strike_put_sold, T, self.risk_free_rate, self.implied_volatility)
        greeks_put_sold = {k: 2 * v for k, v in greeks_put_sold.items()}  # Double the Greeks for the bought option

        # Aggregate Greeks
        return aggregate_greeks({k: -v for k, v in greeks_put_sold.items()}, greeks_put_bought)
    
    def calculate_calendar_spread_with_calls_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculates the greeks for a calendar spread strategy using call options.

        Parameters:
        - S (float): The underlying stock price. If not provided, the last closing price of the stock will be used.
        - T (float): The time until expiration in years. If not provided, the time until expiration will be calculated as the number of days between the expiration date and the current date divided by 365.

        Returns:
        - Dict[str, float]: A dictionary containing the aggregated greeks for the calendar spread strategy with call options. The keys represent the greek names, and the values represent the corresponding greek values.

        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define strikes for the put options
        strike_call_sold = S  # ATM option sold
        strike_call_bought = S   # ITM option bought

        # Greeks for the sold call option
        greeks_near_call_sold = calculate_call_greeks(S, strike_call_sold, T * 0.70, self.risk_free_rate, self.implied_volatility)

        # Greeks for the bought put option
        greeks_far_call_bought = calculate_call_greeks(S, strike_call_bought, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_near_call_sold.items()}, greeks_far_call_bought)
    
    def calculate_calendar_spread_with_puts_greeks(self, S: float = None, T: float = None) -> Dict[str, float]:
        """
        Calculates the greeks for a calendar spread strategy using put options.

        Parameters:
        - S (float): The underlying stock price. If not provided, the last closing price of the stock will be used.
        - T (float): The time until expiration in years. If not provided, the time until expiration will be calculated as the number of days between the expiration date and the current date divided by 365.

        Returns:
        - Dict[str, float]: A dictionary containing the aggregated greeks for the calendar spread strategy with put options. The keys represent the greek names, and the values represent the corresponding greek values.

        """
        if T is None:
            T = (self.expiration_date - datetime.now()).days / 365
        if S is None:
            S = self.stock_data['Close'].iloc[-1]

        # Define strikes for the put options
        strike_put_sold = S  # ATM option sold
        strike_put_bought = S   # ITM option bought

        # Greeks for the sold put option
        greeks_near_put_sold = calculate_put_greeks(S, strike_put_sold, T * 0.70, self.risk_free_rate, self.implied_volatility)

        # Greeks for the bought put option
        greeks_far_put_bought = calculate_put_greeks(S, strike_put_bought, T, self.risk_free_rate, self.implied_volatility)

        return aggregate_greeks({k: -v for k, v in greeks_near_put_sold.items()}, greeks_far_put_bought)

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