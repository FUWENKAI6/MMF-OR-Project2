import numpy as np
import cvxpy as cp
import gurobipy as gp
import pandas as pd
from scipy.stats import gmean

def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2
    return mu, Q

def adjusted_r_squared(e, n_timeperiods, n_factors, SST):
    """
    Calculate the adjusted R-squared given the residuals, number of time periods, 
    number of factors, and total sum of squares (SST).
    """
    SSR = np.sum(e**2)  # Sum of squared residuals across all assets
    R_squared = 1 - (SSR / SST)  # Overall R-squared for the model
    adj_R_squared = 1 - (1 - R_squared) * (n_timeperiods - 1) / (n_timeperiods - n_factors - 1)
    return adj_R_squared

def BSS(returns, factRet, lambda_, K):
    '''
    The BSS model does not use lambda, use a fixed K value
    Arg:
    returns: A pd.DataFrame that contains the returns of 20 assets in time series
    factRet: A pd.DataFrame that contains the returns of 8 factors in time seires
    lambda_: Won't be used in the function
    K: A factor that decides how many factors will be considered

    Return:
    mean: Average predicted mean value for each asset
    cov: covraiance matrix for asset's returns
    r2: return the adjusted r square
    '''
    factRet = np.hstack([np.ones((factRet.shape[0], 1)), factRet.values if isinstance(factRet, np.ndarray) else factRet])
    T, n_factors = factRet.shape
    n_assets = returns.shape[1]
    # Initialize full beta, mu, residuals matrix to store value during iteration
    full_beta_opt = np.zeros((n_factors, n_assets))
    returns = returns.values
    full_predicted_returns = np.zeros_like(returns)
    full_residuals = np.zeros_like(returns)

    for asset_idx in range(0, n_assets):
        single_asset_returns = returns[:, asset_idx]

        model = gp.Model("BSS")
        model.setParam('OutputFlag', True)

        # Define variables
        beta = model.addVars(n_factors, lb=-20, ub=20, name="beta")
        select = model.addVars(n_factors, vtype=gp.GRB.BINARY, name="select")
        # Objective function
        residuals = [single_asset_returns[i] - sum(factRet[i, k] * beta[k] for k in range(n_factors))
                    for i in range(T)]
        model.setObjective(gp.quicksum(res ** 2 for res in residuals), gp.GRB.MINIMIZE)

        # Constraints
        M = 100
        model.addConstr(select.sum() == K, name="LimitFactors")
        for k in range(n_factors):
            model.addConstr(beta[k] <= M * select[k], name=f"UpperLink_{k}")
            model.addConstr(beta[k] >= -M * select[k], name=f"LowerLink_{k}")
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            beta_opt = np.array([beta[k].X for k in range(n_factors)])
            predicted_returns = np.dot(factRet, beta_opt)
            residuals = single_asset_returns - predicted_returns
            full_beta_opt[:,asset_idx] = beta_opt
            full_predicted_returns[:,asset_idx] = predicted_returns
            full_residuals[:,asset_idx] = residuals
        else:
            print("Optimal solution not found.")
    mu = gmean(full_predicted_returns + 1, axis=0) - 1

    # Calculate total sum of squares (SST)
    SST = np.sum((returns - np.mean(returns, axis=0))**2)

    # Construct the diagonal matrix of residual variances
    residual_variances = np.sum(full_residuals**2, axis=0) / (T - n_factors - 1)
    V = full_beta_opt[1:]
    D = np.diag(residual_variances)
    # Calculate the covariance matrix of the factor model
    F = np.cov(factRet[:,1:], rowvar=False)  # Factor covariance matrix
    Q = np.dot(V.T, np.dot(F, V)) + D  # Total variance-covariance matrix including residuals

    adj_R_squared = adjusted_r_squared(full_residuals, T, n_factors, SST)
    return mu, Q, adj_R_squared

def FF(returns, factRet, lambda_=None, K=None):
    """
    Calibrate the Fama-French 3-factor model using the Market, Size, and Value factors.
    """
    # Ensure input is numpy arrays
    if isinstance(returns, pd.DataFrame):
        returns = returns.values
    if isinstance(factRet, pd.DataFrame):
        # Extract only the three FF factors: Market, Size, and Value
        factRet = factRet[['Mkt_RF', 'SMB', 'HML']].values

    # Get dimensions of the returns and factors
    n_timeperiods, n_assets = returns.shape
    n_factors = factRet.shape[1]  # Should be 3 for the FF model

    # Prepare the design matrix X with an intercept
    X = np.hstack([np.ones((n_timeperiods, 1)), factRet])

    # Perform OLS regression to find beta coefficients
    B = np.linalg.inv(X.T @ X) @ X.T @ returns  # ensure B is a numpy array

    # Slice out the intercept (alpha) and betas
    alphas = B[0]
    V = B[1:]

    # Calculate expected returns using the average factor returns
    f_bar = gmean(factRet+1, axis=0)-1
    mu = alphas + np.dot(V.T, f_bar)

    # Calculate residuals
    e = returns - X @ B
    SST = np.sum((returns - np.mean(returns, axis=0))**2)


    # Calculate unbiased estimates of residual variances
    residual_variances = np.sum(e**2, axis=0) / (n_timeperiods - n_factors - 1)

    # Construct the diagonal matrix of residual variances
    D = np.diag(residual_variances)

    # Calculate the covariance matrix of the factor model
    F = np.cov(factRet, rowvar=False)  # Factor covariance matrix
    Q = np.dot(V.T, np.dot(F, V)) + D  # Total variance-covariance matrix including residuals
    adj_R_squared = adjusted_r_squared(e, n_timeperiods, n_factors, SST)

    return mu, Q, adj_R_squared

def LASSO(returns, factRet, lambda_, K=None):
    """
    The LASSO model does not use K, a lambda is used
    Arg:
    returns: A pd.DataFrame that contains the returns of 20 assets in time series
    factRet: A pd.DataFrame that contains the returns of 8 factors in time seires
    lambda_: The penalty for L1 regularization
    K: Won't be used in this function

    Return:
    mean: Average predicted mean value for each asset
    cov: covraiance matrix for asset's returns
    r2: return the adjusted r square
    """
    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------
    factRet = np.hstack([np.ones((factRet.shape[0], 1)), factRet.values])
    T, n_assets = returns.shape
    n_factors = factRet.shape[1]
    returns = returns.values

    # Initialize matrix
    beta = cp.Variable((n_factors, n_assets))
    residuals = returns - (factRet @ beta)
    lasso_penalty = lambda_ * cp.norm(beta,1)

    # Build the objective function using residuals and penalties
    objective = cp.Minimize(cp.sum_squares(residuals) + lasso_penalty)
    problem = cp.Problem(objective)
    problem.solve()

    # Calculate expected asset return
    # mu = np.mean( factRet @ beta.value, axis=0)
    mu = gmean( factRet @ beta.value + 1, axis=0)-1

    residuals = returns - factRet @ beta.value

    # Calculate total sum of squares (SST)
    SST = np.sum((returns - np.mean(returns, axis=0))**2)

    # Calculate unbiased estimates of residual variances
    residual_variances = np.sum(residuals**2, axis=0) / (T - n_factors - 1)

    # Construct the diagonal matrix of residual variances
    V = beta.value[1:]
    D = np.diag(residual_variances)
    # Calculate the covariance matrix of the factor model
    F = np.cov(factRet[:,1:], rowvar=False)  # Factor covariance matrix
    Q = np.dot(V.T, np.dot(F, V)) + D  # Total variance-covariance matrix including residuals

    adj_R_squared = adjusted_r_squared(residuals, T, n_factors, SST)
    return mu, Q, adj_R_squared

if __name__ == '__main__':
    print(f"{'-'*16} \nRun Successfully")