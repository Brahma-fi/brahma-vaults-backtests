import numpy as np
from scipy.stats import norm 
 
def option_payoff(S,K,flag):
    
    payoff = max(flag*(S-K),0)
    
    return payoff

def black_scholes_price(S,K,T,sigma,r,flag):
    
    d1 = ( np.log(S/K)+(r+sigma**2/2) * T ) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    price = flag*S*norm.cdf(flag*d1)-flag*K*np.exp(-r*T)*norm.cdf(flag*d2)
    return price