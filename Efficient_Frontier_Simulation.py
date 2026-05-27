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
import matplotlib.pyplot as plt

TRADING_DAYS = 252
SIMULATIONS = 100000

"""
Check if number of positions is valid and then run simulation on portfolio.
"""
def main():
    # get positions in portfolio from user
    positions = getPortfolio()

    # catch if there is no positions given
    if (len(positions) < 1):
        print("No positions given")
    else:
        # get historical data and run simulation
        historical_price_data = retrieveHistoricalData(positions)
        annualized_returns, corr_matrix, cov_matrix, rf = MCSInputs(historical_price_data)
        randomized_weights, mcs_results = monteCarloSimulation(
            positions, 
            annualized_returns, 
            cov_matrix,
            rf
            )
        displayMCS(
            positions, 
            corr_matrix, 
            cov_matrix, 
            randomized_weights, 
            mcs_results
            )

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
    # calculate simple returns for each day and clean (first day will be NaN)
    simple_returns = (historical_price_data / historical_price_data.shift(1)) - 1
    cleaned_returns = simple_returns.dropna()
    return cleaned_returns

"""
Calculates the annualized return of each position. This is calculated by taking
the 10 year compound annual growth rate (CAGR).
"""
def annualizedReturnCalculation(historical_price_data):
    # calculate the change and the factor, return annualized returns
    change = (historical_price_data.iloc[-1] / historical_price_data.iloc[0])
    annualization_factor = (1 / ((len(historical_price_data) - 1) / TRADING_DAYS))
    annualized_returns = (change ** annualization_factor) - 1
    return annualized_returns

"""
Calculates the annualized volatility of each position.
"""
def volatilityCalculation(simple_returns):
    # get the standard deviation of simple daily returns
    standard_deviation = simple_returns.std()

    # annualize the standard deviation to get the annualized volatility
    sigma = standard_deviation * np.sqrt(TRADING_DAYS)
    return sigma

"""
Correlation measures the degree to which two equities move in lock-step with one
another. Their correlation value can range from -1.0 (inversely correlated) to
1.0 (positively correlated). The correlation matrix is calculated by taking
simple returns of each equity in the portfolio and computing the pairwise
correlation coefficients between all equity pairs.
"""
def correlationCalculation(simple_returns):
    # calculate the correlation of each asset pair
    corr_matrix = np.array(simple_returns.corr())
    return corr_matrix

"""
Covariance measures how much two equities move together, accounting for their
individual volatilities and the correlation of the asset pair. An equity with 
high volatility that is highly correlated to another equity will have a large 
covariance with that equity, whereas uncorrelated equities will have a 
covariance near zero. The covariance matrix is simply the correlation matrix 
scaled by the volatilities of each asset pair.
"""
def covarianceCalculation(sigma, corr_matrix):
    # multiply the correlation matrix by the vector of volatilities
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return cov_matrix

"""
This function calculates all needed inputs for Monte Carlo simulation.
"""
def MCSInputs(historical_price_data):
    annualized_returns = annualizedReturnCalculation(historical_price_data)
    simple_returns = dailyReturnCalculation(historical_price_data)
    sigma = volatilityCalculation(simple_returns)
    corr_matrix = correlationCalculation(simple_returns)
    cov_matrix = covarianceCalculation(sigma, corr_matrix)
    rf = (yf.download("^TNX", period="5d", auto_adjust=True)["Close"].iloc[-1]) / 100
    rf = float(rf.iloc[0])

    return annualized_returns, corr_matrix, cov_matrix, rf

"""
Monte Carlo simulation performed by generating a large number of portfolios
through randomized position weight combinations. For each simulation, random 
numbers are generated for each position and then normalized to sum to 1 in order
to make sure all capital is utilized and there are no short positions. The
return, volatility, and Sharpe ratio of the portfolio are then calculated based
on each weight combination.
"""
def monteCarloSimulation(positions, annualized_returns, cov_matrix, rf):
    # create array to house randomly generated numbers
    random_nums = np.random.random((SIMULATIONS, len(positions)))
    # sum each simulations random numbers
    weight_sums = np.sum(random_nums, axis=1)
    # divide each random number by the sum of the simulation to get the weight
    randomized_weights = random_nums / weight_sums.reshape(SIMULATIONS, 1)

    # calculate the return by weighting the annualized returns of each position
    portfolio_return = randomized_weights @ np.array(annualized_returns)
    # calculate the variance by applying each weight vector to the covariance
    variance = np.array([w @ cov_matrix @ w for w in randomized_weights])
    volatility = np.sqrt(variance)
    # calculate the sharpe of the portfolio
    sharpe = (portfolio_return - rf) / volatility

    # stack the results into one data frame
    # columns are the portfolio metrics and rows are simulations
    mcs_results = np.column_stack([portfolio_return, volatility, sharpe])

    return randomized_weights, mcs_results

"""
This function generates the graphical display of the portfolio.
"""
def displayMCS(positions, corr_matrix, cov_matrix, randomized_weights, mcs_results):
    # create new figure
    fig = plt.figure(figsize=(15, 10))
    # add title
    fig.suptitle("Efficient Frontier Simulation", fontsize=22, weight='bold')

    # adding axes to figure for display
    top_left = fig.add_axes([0.06, 0.53, 0.40, 0.40])
    top_left.axis('off')
    bottom_left = fig.add_axes([0.06, 0.06, 0.40, 0.40])
    top_right = fig.add_axes([0.56, 0.53, 0.40, 0.40])
    bottom_right = fig.add_axes([0.56, 0.06, 0.40, 0.40])

    # displaying efficient frontier
    scatter = bottom_left.scatter(
        mcs_results[:, 1], # volatility on x-axis
        mcs_results[:, 0], # return on y-axis
        c = mcs_results[:, 2], # color by sharpe ratio
        cmap='viridis',
        alpha=0.5,
        s=10
    )

    # display asset correlation matrix
    

    plt.show()


if __name__=="__main__":
    main()