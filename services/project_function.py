import sys
sys.path.append('/Users/fuwenkai/Documents/U of T/MMF/MMF 1921 Operations Research/MMF Project 2/Code')
# Delete above line if necessary
from services.strategies import *

def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = OLS_MVO()
    x, mu = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x, mu # remove mu after develop

