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
import seaborn as sns
import mplcursors

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
    # adding single position portfolios
    single_position_ports = np.identity(len(positions))
    randomized_weights = np.vstack([single_position_ports, randomized_weights])

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
    top_left = fig.add_axes([0.08, 0.52, 0.38, 0.38])
    top_left.axis('off')
    bottom_left = fig.add_axes([0.08, 0.06, 0.38, 0.38])
    top_right = fig.add_axes([0.58, 0.52, 0.38, 0.38])
    bottom_right = fig.add_axes([0.58, 0.06, 0.38, 0.38])

    # displaying efficient frontier
    scatter = bottom_left.scatter(
        mcs_results[:, 1], # volatility on x-axis
        mcs_results[:, 0], # return on y-axis
        c = mcs_results[:, 2], # color by sharpe ratio
        cmap='viridis',
        s=10
    )
    bottom_left.set_title("Efficient Frontier of Portfolios", fontsize=14)
    bottom_left.set_xlabel("Volatility")
    bottom_left.set_ylabel("Return")
    plt.colorbar(scatter, ax=bottom_left)

    # getting single position portfolios to mark on efficient frontier
    for p in range(0, len(positions)):
        # get index of portfolio with full allocation of each position
        index_single_position_port = np.argmax(randomized_weights[:, p])
        bottom_left.scatter(
            mcs_results[index_single_position_port, 1],
            mcs_results[index_single_position_port, 0],
            marker='x',
            color='black',
            s=50,
            zorder=5,
            label=positions[p]
            )

    # add label for highest sharpe portfolio text box
    top_left.text(0.10, 1, "Highest Sharpe", fontsize=14)

    # add label for minimum variance portfolio text box
    top_left.text(0.60, 1, "Minimum Variance", fontsize=14)

    # display portfolio with highest sharpe and its metrics
    # get metrics of highest sharpe portfolio
    index_highest_sharpe = np.argmax(mcs_results[:, 2])
    weights = randomized_weights[index_highest_sharpe]
    portfolio_return = mcs_results[index_highest_sharpe, 0]
    volatility = mcs_results[index_highest_sharpe, 1]
    sharpe = mcs_results[index_highest_sharpe, 2]
    # compile metrics to formated text
    portfolio_metrics = (
        f"Return: {portfolio_return:.2%}\n" # display return of portfolio
        f"Volatility: {volatility:.2%}\n" # display volatility of portfolio
        f"Sharpe: {sharpe:.2}\n\n" # display sharpe of portfolio
        "Position Weights:\n" # display weight of each position in portfolio
        "--------------------\n" +
        "\n".join([f"{positions[i]}: {weights[i]:.2%}"
                    for i in range(0, len(positions))])
    )
    # format text box and text
    top_left.text(
        0.025, 
        0.96, 
        portfolio_metrics, # text to display
        fontsize=20, 
        verticalalignment='top',
        horizontalalignment='left',
        color='black',
        linespacing=1.5,
        bbox=dict(facecolor='#FBE5D6', edgecolor='black', boxstyle='square')
        )
    # display highest sharpe portfolio on efficient frontier
    bottom_left.scatter(
        mcs_results[index_highest_sharpe, 1],
        mcs_results[index_highest_sharpe, 0],
        marker='*',
        color='black',
        s=75,
        zorder=5,
        label='Max Sharpe'
        )

    # display portfolio with lowest variance and its metrics
    # get metrics of lowest variance portfolio
    index_min_variance = np.argmin(mcs_results[:, 1])
    weights = randomized_weights[index_min_variance]
    portfolio_return = mcs_results[index_min_variance, 0]
    volatility = mcs_results[index_min_variance, 1]
    sharpe = mcs_results[index_min_variance, 2]
    # compile metrics to formated text
    portfolio_metrics = (
        f"Return: {portfolio_return:.2%}\n" # display return of portfolio
        f"Volatility: {volatility:.2%}\n" # display volatility of portfolio
        f"Sharpe: {sharpe:.2}\n\n" # display sharpe of portfolio
        "Position Weights:\n" # display weight of each position in portfolio
        "--------------------\n" +
        "\n".join([f"{positions[i]}: {weights[i]:.2%}"
                    for i in range(0, len(positions))])
    )
    # format text box and text
    top_left.text(
        0.55, 
        0.96, 
        portfolio_metrics, # text to display
        fontsize=20, 
        verticalalignment='top',
        horizontalalignment='left',
        color='black',
        linespacing=1.5,
        bbox=dict(facecolor='#FBE5D6', edgecolor='black', boxstyle='square')
        )
    # display minimum variance portfolio on efficient frontier
    bottom_left.scatter(
        mcs_results[index_min_variance, 1],
        mcs_results[index_min_variance, 0],
        marker='D',
        color='black',
        s=50,
        zorder=5,
        label='Min Variance'
        )

    # display asset correlation matrix
    sns.heatmap(
        corr_matrix, # heatmap of correlations
        annot=True,
        cmap='RdYlGn',
        xticklabels=positions, # display tickers on x
        yticklabels=positions, # display tickers on y
        fmt=".2f", 
        ax=top_right
        )
    top_right.set_title("Correlation Matrix", fontsize=14)
    
    # display asset covariance matrix
    sns.heatmap(
        cov_matrix, # heatmap of correlations
        annot=True,
        cmap='coolwarm',
        xticklabels=positions, # display tickers on x
        yticklabels=positions, # display tickers on y
        fmt=".2f", 
        ax=bottom_right
        )
    bottom_right.set_title("Covariance Matrix", fontsize=14)

    # adding mplcursors functionality so hovering over portfolio shows details
    cursor = mplcursors.cursor(scatter, hover=True)

    # connect hover event
    @cursor.connect("add")
    # when hover event
    def on_add(selected_portfolio):
        # get index of portfolio and determine portfolio metrics
        index = selected_portfolio.index
        weights = randomized_weights[index]
        portfolio_return = mcs_results[index, 0]
        volatility = mcs_results[index, 1]
        sharpe = mcs_results[index, 2]

        # display annotation on portfolio
        selected_portfolio.annotation.set_text(
            f"Return: {portfolio_return:.2%}\n" # display return of portfolio
            f"Volatility: {volatility:.2%}\n" # display volatility of portfolio
            f"Sharpe: {sharpe:.2}\n\n" # display sharpe of portfolio
            "Position Weights:\n" # display weight of each position in portfolio
            "--------------------\n" +
            "\n".join([f"{positions[i]}: {weights[i]:.2%}"
                       for i in range(0, len(positions))])
        )

        # format annotation box
        selected_portfolio.annotation.get_bbox_patch().set(
            facecolor='white', 
            alpha=0.8,
            boxstyle='round'
            )
        
        # format annotation text
        selected_portfolio.annotation.set_color('black')

    bottom_left.legend(fontsize=8, markerscale=0.6)
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()