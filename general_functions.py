import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
from typing import Dict, List

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a call option using the Black-Scholes formula.

    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset

    Returns:
    float: Price of the call option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def calculate_call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate the Greeks for a call option.

    Parameters:
    S, K, T, r, sigma: Refer to the Black-Scholes parameters

    Returns:
    dict: Greeks of the call option (Delta, Gamma, Theta, Vega, Rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a put option using the Black-Scholes formula.

    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset

    Returns:
    float: Price of the put option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price

def calculate_put_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate the Greeks for a put option.

    Parameters:
    S, K, T, r, sigma: Refer to the Black-Scholes parameters

    Returns:
    dict: Greeks of the put option (Delta, Gamma, Theta, Vega, Rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

def aggregate_greeks(*greeks_lists: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregate the Greeks for multiple options.

    Parameters:
    greeks_lists (list of dicts): List of Greeks for each option in the spread.

    Returns:
    dict: Aggregated Greeks for the spread.
    """
    aggregated_greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    for greeks in greeks_lists:
        for key in aggregated_greeks:
            aggregated_greeks[key] += greeks[key]
    return aggregated_greeks