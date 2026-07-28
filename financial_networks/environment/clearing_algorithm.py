import numpy as np

def compute_net_worth(liabilities, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, price_risky_asset, payment):
    '''
    Computes the net worth of banks.
    
    Parameters:
      - liabilities (np.array): Vector of payment obligations.
      - liabilities_matrix (np.array): Matrix of relative liabilities between banks.
      - shares_risk_free_asset (np.array): Vector of risk-free asset shares.
      - shares_risky_asset (np.array): Vector of risky asset shares.
      - price_risky_asset (float): Current price of the risky asset.
      - payment (np.array): Vector of realized payments.
      
    Returns:
      - net_worth (np.array): Vector of the net worth of banks.
    '''

    claims = liabilities_matrix.T @ payment
    net_worth = (shares_risk_free_asset + shares_risky_asset * price_risky_asset - liabilities + claims)
    return net_worth
    
def maximal_fixed_point(payments, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, price_risky_asset, insolvent_banks, alpha, beta, inverse_demand_function, tolerance=1e-4, max_iter=100000):
    '''
    Compute clearing payments for a given default set by fixed-point iteration.

    The iteration follows Weber and Weske, Algorithm 3.4, adapted to our
    financial-network model.
    Only insolvent banks liquidate risky assets.
    
    Parameters:
      - payments (np.array): Vector of payments.
      - liabilities_matrix (np.array): Matrix of relative liabilities between banks.
      - shares_risk_free_asset (np.array): Vector of risk-free asset shares.
      - shares_risky_asset (np.array): Vector of risky asset shares.
      - price_risky_asset (float): Price of the risky asset.
      - insolvent_banks (set): Set of insolvent banks.
      - alpha (float): Fraction of external asset value realized by a defaulting bank (0 < alpha <= 1).
      - beta (float): Fraction of interbank asset value realized by a defaulting bank (0 < beta <= 1).
      - inverse_demand_function (function): Inverse demand function for computation of the new price.
      - tolerance (float): Tolerance for the convergence condition.
      - max_iter (int): Maximum number of iterations.
      
    Returns:
      - x (np.array): Payment vector for the given insolvent banks.
      - y (float): New price of the risky asset.
    '''
    
    n = len(shares_risk_free_asset)
    x = payments.copy()
    y_0 = float(price_risky_asset)
    solvent_banks = set(range(n)) - insolvent_banks
    # Update price: as only insolvent banks sell shares, the new price can be calculated directly
    if y_0 > 0:
        y = y_0 * inverse_demand_function(sum(shares_risky_asset[i] for i in insolvent_banks))
    elif y_0 == 0:
        y = -1e-8 * inverse_demand_function(-sum(shares_risky_asset[i] for i in insolvent_banks)*1e-6)
    elif y_0 < 0:
        y = y_0 * inverse_demand_function(-sum(shares_risky_asset[i] for i in insolvent_banks)*1e-6)
    
    for _ in range(max_iter):
        x_new = x.copy()
        
        # Update payment vector for insolvent banks
        for i in insolvent_banks:
            # Apply the recovery factors to asset values and incoming payments.
            x_new[i] = np.max(alpha * (shares_risk_free_asset[i] + shares_risky_asset[i] * y) + beta * (liabilities_matrix[:, i] @ x),0)

        # Check convergence
        diff = np.max(np.abs(x_new - x))
        if diff < tolerance:
            return x_new, y
            
        x = x_new
        
    # If convergence is not achieved within the maximum number of iterations, return the last computed values
    print("Warning: Maximal fixed point not found within the maximum number of iterations.")
    print(f" Difference in last iteration: {diff}")
    return x, y  

def equilibrium_algorithm(liabilities, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, initial_price_risky_asset, alpha, beta, inverse_demand_function, tolerance=1e-4, max_iter=100000):    
    '''
    Compute equilibrium payments, the risky-asset price, and bank net worth.

    The default-set iteration follows Weber and Weske, Algorithm 3.4.
    
    Parameters:
      - liabilities (np.array): Vector of total payment obligations.
      - liabilities_matrix (np.array): Matrix of relative liabilities between banks.
      - shares_risk_free_asset (np.array): Vector of risk-free asset shares.
      - shares_risky_asset (np.array): Vector of risky asset shares.
      - initial_price_risky_asset (float): Initial price of the risky asset.
      - alpha (float): Fraction of external asset value realized by a defaulting bank (0 < alpha <= 1).
      - beta (float): Fraction of interbank asset value realized by a defaulting bank (0 < beta <= 1).
      - inverse_demand_function (function): Inverse demand function for computation of the new price.
      - tolerance (float): Tolerance for the convergence condition.
      - max_iter (int): Maximum number of iterations.
      
    Returns:
      - p (np.array): Equilibrium payment vector.
      - q (float): Equilibrium price of the risky asset.
      - w (np.array): Net worth of the banks.
    '''

    # Step 0 in Algorithm 3.4
    n = len(shares_risk_free_asset)
    D_prev = set()
    w = compute_net_worth(liabilities, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, initial_price_risky_asset, liabilities)
    D = {i for i in range(n) if w[i] < -1e-8}
    p = liabilities.copy()  # Payment vector
    q = initial_price_risky_asset
    
    while True:
        # Step 1 in Algorithm 3.4
        if D == D_prev:
            break
        # Step 2 in Algorithm 3.4
        D_prev = D
        p, q = maximal_fixed_point(p, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, q, D_prev, alpha, beta, inverse_demand_function, tolerance, max_iter)
        w = compute_net_worth(liabilities, liabilities_matrix, shares_risk_free_asset, shares_risky_asset, q,p)
        D = {i for i in range(n) if w[i] < - tolerance}  
    
    return p, q, w,

