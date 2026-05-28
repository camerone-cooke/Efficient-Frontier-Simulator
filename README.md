# Efficient Frontier Simulator

Efficient Frontier simulator using Monte Carlo simulation and multi-asset covariance for generating a modern portfolio theory analytical dashboard

---

## Project Synopsis

This program simulates potential weight allocations of an equity portfolio using Monte Carlo simulation to generate the Efficient Frontier. Live market data is dynamically pulled from Yahoo Finance via the yfinance library to compute annualized equity returns, historical volatilities, and correlations between asset pairs. The highly optimized simulation generates 100,000+ portfolios with differing position allocations to identify both the Maximum Sharpe Ratio portfolio (generates the highest return per unit of risk) and the Minimum Variance portfolio. The resulting multi-panel display functions as a PM's dashboard, providing correlation and covariance heatmaps, a precise breakdown of the Maximum Sharpe and Minimum Variance portfolios, as well as an interactive Efficient Frontier. Noteworthy portfolios are also highlighted along with a representation of the Capital Market Line which allows users to explore the theoretical trade-offs of risk and return across portfolios with varying asset structures.

