#!/usr/bin/env python
# coding: utf-8

# # quant-finance-model-comparison

# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model


# In[2]:


# ==============================
# CONFIG
# ==============================

TICKER = "HRHO.CA"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
FORECAST_DATE = "2024-04-01"

np.random.seed(42)


# In[3]:


# ==============================
# LOAD DATA
# ==============================

def load_data(ticker):
    data = yf.download(ticker, period="max")
    data["Price"] = data["Adj Close"] if "Adj Close" in data else data["Close"]
    return data


# In[4]:


# ==============================
# GBM MODEL
# ==============================

def estimate_gbm(prices):
    lr = np.log(prices / prices.shift(1)).dropna()
    mu = lr.mean() * 252
    sigma = lr.std() * np.sqrt(252)
    return mu, sigma

def simulate_gbm(S0, mu, sigma, days=63, paths=5000):
    dt = 1/252
    Z = np.random.normal(size=(paths, days))
    S = np.zeros((paths, days))
    S[:,0] = S0

    for t in range(1, days):
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t])

    return S.mean(axis=0)


# In[5]:


# ==============================
# HESTON MODEL
# ==============================

def estimate_heston(prices):
    lr = np.log(prices / prices.shift(1)).dropna()
    rv = lr.rolling(10).var().dropna()

    kappa = 1.0
    theta = rv.mean()
    xi = 0.3
    rho = -0.4
    v0 = rv.iloc[0]
    mu = lr.mean() * 252

    return kappa, theta, xi, rho, v0, mu

def simulate_heston(S0, params, days=63, paths=5000):
    kappa, theta, xi, rho, v0, mu = params
    dt = 1/252

    S = np.zeros((paths, days))
    v = np.zeros((paths, days))

    S[:,0] = S0
    v[:,0] = v0

    for t in range(1, days):
        z1 = np.random.normal(size=paths)
        z2 = np.random.normal(size=paths)

        dWv = z1
        dWs = rho*z1 + np.sqrt(1-rho**2)*z2

        v[:,t] = np.maximum(v[:,t-1] + kappa*(theta-v[:,t-1])*dt + xi*np.sqrt(v[:,t-1])*np.sqrt(dt)*dWv, 1e-8)
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*v[:,t-1])*dt + np.sqrt(v[:,t-1])*np.sqrt(dt)*dWs)

    return S.mean(axis=0)


# In[6]:


# ==============================
# GARCH MODEL
# ==============================

def estimate_garch(prices):
    returns = 100 * prices.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')

    sigma = res.conditional_volatility / 100
    mu = returns.mean() / 100
    return sigma, mu

def simulate_garch_gbm(S0, mu, sigma_series, days, paths=5000):
    dt = 1/252
    S = np.zeros((paths, days))
    S[:,0] = S0

    for t in range(1, days):
        sigma_t = sigma_series.iloc[min(t-1, len(sigma_series)-1)]
        Z = np.random.normal(size=paths)
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*sigma_t**2)*dt + sigma_t*np.sqrt(dt)*Z)

    return S.mean(axis=0)


# In[7]:


# ==============================
# MAIN PIPELINE
# ==============================

def run_models():
    data = load_data(TICKER)
    train = data["Price"].loc[START_DATE:END_DATE]

    S0 = train.iloc[-1]

    future_dates = data.loc["2024-01-01":"2024-04-01"].index
    horizon = len(future_dates)

    # === GBM ===
    mu_gbm, sigma_gbm = estimate_gbm(train)
    gbm_path = simulate_gbm(S0, mu_gbm, sigma_gbm, horizon)

    # === HESTON ===
    heston_params = estimate_heston(train)
    heston_path = simulate_heston(S0, heston_params, horizon)

    # === GARCH ===
    sigma_series, mu_garch = estimate_garch(train)
    garch_path = simulate_garch_gbm(S0, mu_garch, sigma_series, horizon)

    # === Actual ===
    actual = data["Price"].loc["2024-01-01":"2024-04-01"]

    # ==============================
    # PLOT
    # ==============================

    plt.figure(figsize=(12,6))

    plt.plot(future_dates, gbm_path, label="GBM")
    plt.plot(future_dates, heston_path, label="Heston")
    plt.plot(future_dates, garch_path, label="GARCH-GBM")

    plt.title(f"Model Comparison - {TICKER}")
    plt.xlabel("Date")
    plt.ylabel("Percentage Error (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_models()


# In[ ]:




