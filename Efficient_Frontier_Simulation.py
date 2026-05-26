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
import numpy as np

"""
Check if number of positions is valid and then run simulation on portfolio.
"""
def main():
    positions = getPortfolio()
    if (len(positions) < 1):
        print("No positions given")
    else:
        return

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


if __name__=="__main__":
    main()