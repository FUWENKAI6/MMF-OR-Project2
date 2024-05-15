import cvxpy as cp
import numpy as np
from scipy.stats import chi2
# 1. CVAR - Kai
# 2. Robust MVO - Grace
# 3. MVO - Given
# 4. Risk Parity - Jay
def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value

def robustMVO(mu, Q, lambda_=0.02, alpha=0.95, T=20):
    """
    Construct a robust MVO portfolio considering uncertainties in mu and Q.
    Args:
        mu (np.ndarray): Expected returns.
        Q (np.ndarray): Covariance matrix.
        lambda_ (float): Risk aversion coefficient.
        alpha (float): Confidence level.
        T (int): Number of return observations.
    Returns:
        np.ndarray: Optimal portfolio weights.
    """
    # Number of assets
    n = len(mu)
    
    # Radius of the uncertainty set
    ep = np.sqrt(chi2.ppf(alpha, n))
    
    # Squared standard error of expected returns
    Theta = np.diag(np.diag(Q)) / T
    
    # Square root of Theta
    sqrtTh = np.sqrt(Theta)
    
    # Initial portfolio (equally weighted)
    x0 = np.ones(n) / n
    
    # Variables
    x = cp.Variable(n)
    
    # Objective function
    objective = cp.Minimize(lambda_ * cp.quad_form(x, Q) - mu.T @ x + ep * cp.norm(sqrtTh @ x))
    
    # Constraints
    constraints = [
        cp.sum(x) == 1,  # Sum of weights equals 1
        x >= 0           # No short sales
    ]
    
    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return x.value

def CVaR(mu, Q, returns, alpha=0.95):
    T, n = returns.shape  # T: number of periods, n: number of assets

    # Define variables
    w = cp.Variable(n)  # Portfolio weights
    VaR = cp.Variable()  # Value at Risk
    u = cp.Variable(T)  # Auxiliary variables

    # Define constraints
    constraints = [
        cp.sum(w) == 1,  # Sum of weights equals 1 (no leverage)
        w >= 0,  # No short selling
        u >= 0,  # Auxiliary variables must be non-negative
        u >= -returns @ w - VaR  # CVaR constraint
    ]

    # Define the objective function
    objective = cp.Minimize(VaR + (1 / (T * (1 - alpha))) * cp.sum(u))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal portfolio weights
    return w.value
