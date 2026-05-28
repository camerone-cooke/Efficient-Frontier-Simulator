# Efficient Frontier Simulator

Efficient Frontier simulator using Monte Carlo simulation and multi-asset covariance for generating a modern portfolio theory analytical dashboard

---

## Project Synopsis

This program simulates potential weight allocations of an equity portfolio using Monte Carlo simulation to generate the Efficient Frontier. Live market data is dynamically pulled from Yahoo Finance via the yfinance library to compute annualized equity returns, historical volatilities, and correlations between asset pairs. The highly optimized simulation generates 100,000+ portfolios with differing position allocations to identify both the Maximum Sharpe Ratio portfolio (generates the highest return per unit of risk) and the Minimum Variance portfolio. The resulting multi-panel display functions as a PM's dashboard, providing correlation and covariance heatmaps, a precise breakdown of the Maximum Sharpe and Minimum Variance portfolios, as well as an interactive Efficient Frontier. Noteworthy portfolios are also highlighted along with a representation of the Capital Market Line which allows users to explore the theoretical trade-offs of risk and return across portfolios with varying asset structures.

---

## Technical Approach

### Monte Carlo Simulation

#### What is Monte Carlo Simulation?

Monte Carlo simulation is a way to model the probability of different outcomes for a process whose outcome cannot be easily predicted due to the inclusion of random variables. Monte Carlo simulation is used to understand the impact of risk and uncertainty and in investing can be used to model the range of potential risk and return outcomes across different portfolio weight allocations.

#### Monte Carlo Simulation Used in This Project

In this project, Monte Carlo simulation is being used to generate hundreds of thousands of portfolios with differing position allocations. Utilizing a Monte Carlo simulation allows investors to better grasp the range of possible risk-return outcomes and identify optimal distributions of capital. This allows for the estimation of the Maximum Sharpe Ratio portfolio and the calculation of the Minimum Variance portfolio.



