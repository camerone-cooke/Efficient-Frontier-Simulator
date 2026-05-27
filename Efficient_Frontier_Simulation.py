# Cameron Cooke
# Copyright © 2026

"""
This program utilizes historical equity returns and a benchmark tracking error
constraint to simulate and visualize the efficient frontier. Using a Monte Carlo
simulation, the system maps thousands of distinct portfolios with varying asset
weight combinations to a risk-return scatter plot. This visualization displays 
the risk-return tradoff, allowing for the identitfication of optimal position 
allocations that maximize the return for a given level of risk.
"""

# import needed libraries
import yfinance as yf
import numpy as np

TRADING_DAYS = 252

"""
Check if number of positions is valid and then run simulation on portfolio.
"""
def main():
    positions = getPortfolio()
    if (len(positions) < 1):
        print("No positions given")
    else:
        historical_price_data = retrieveHistoricalData(positions)

"""
Prompt user for positions in portfolio.
"""
def getPortfolio():
    positions = np.array([])

    # prompt user for ticker
    ticker = input('What Equity\'s price would you like to simulate? '
                    'or \'quit\' to stop: ').upper()
    while (ticker != "QUIT"):
        # add ticker to positions and number of shares to shares
        positions = np.append(positions, ticker)

        # re-prompt user for next ticker
        ticker = input('What Equity\'s price would you like to simulate? '
                    'or \'quit\' to stop: ').upper()
        
    return positions

"""
Retrieves historical data from positions in portfolio.
"""
def retrieveHistoricalData(positions):
    # use yf.download instead of yf.Ticker (for a single ticker) or yf.Tickers 
    # (for multiple tickers) due to being more efficient (uses multi-threading)

    # retrieve data for all positions at one time and place in dataframe
    historical_price_data = yf.download(
        positions.tolist(), 
        period="10y", 
        auto_adjust=True
        )["Close"]
    
    return historical_price_data

"""
Calculates the daily return of each position
"""
def dailyReturnCalculation(historical_price_data):
    simple_returns = (historical_price_data / historical_price_data.shift(1)) - 1
    cleaned_returns = simple_returns.dropna()
    return cleaned_returns

"""
Calculates the annualized return of each position. This is calculated by taking
the 10 year compound annual growth rate (CAGR).
"""
def annualizedReturnCalculation(historical_price_data):
    change = (historical_price_data.iloc[-1] / historical_price_data.iloc[0])
    annualization_factor = (1 / ((len(historical_price_data) - 1) / TRADING_DAYS))
    annualized_returns = (change ** annualization_factor) - 1
    return annualized_returns

"""
Calculates the annualized volatility of each position.
"""
def volatilityCalculation(simple_returns):
    standard_deviation = simple_returns.std()
    sigma = standard_deviation * np.sqrt(TRADING_DAYS)
    return sigma

"""
Correlation measures the degree to which two equities move in lock-step with one
another. Their correlation value can range from -1.0 (inversely correlated) to
1.0 (positively correlated). The correlation matrix is calculated by taking
simple returns of each equity in the portfolio and computing the pairwise
correlation coefficients between all equity pairs.
"""
def correlationCalculation(historical_price_data):
    simple_returns = dailyReturnCalculation(historical_price_data)
    corr_matrix = np.array(simple_returns.corr())
    return corr_matrix


if __name__=="__main__":
    main()