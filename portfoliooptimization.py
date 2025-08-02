from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# Set Streamlit page config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ----------------------- Function: Download Price Data -----------------------
@st.cache_data
def download_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data.dropna()

# ----------------------- Function: Calculate Returns -----------------------
def calculate_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns

# ----------------------- Function: Filter ESG Stocks (Placeholder logic) -----------------------
def esg_filter(tickers):
    # Placeholder ESG logic: just exclude one for demonstration
    return [ticker for ticker in tickers if ticker != "XOM"]

# ----------------------- Function: Cluster Stocks -----------------------
def cluster_stocks(returns, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(returns.T)
    clusters = pd.Series(model.labels_, index=returns.columns, name="Cluster")
    return clusters

# ----------------------- Function: Portfolio Optimization -----------------------
def optimize_portfolio(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)

    def portfolio_performance(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return -sharpe  # maximize Sharpe => minimize -Sharpe

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    result = minimize(portfolio_performance, init_guess, bounds=bounds, constraints=constraints)
    return result.x

# ----------------------- Function: Plot Efficient Frontier -----------------------
def plot_efficient_frontier(returns, risk_free_rate=0.02, n_portfolios=5000):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)

    results = np.zeros((3, n_portfolios))
    weight_array = []

    for i in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_array.append(weights)
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_vol
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    opt_vol, opt_return = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap="viridis", alpha=0.5)
    ax.scatter(opt_vol, opt_return, c="red", marker="*", s=100, label="Optimal")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    plt.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

# ----------------------- Main Streamlit App -----------------------
def run_app():
    st.title(" Portfolio Optimization with ESG & Clustering")

    # Sidebar Inputs
    tickers = st.sidebar.multiselect("Choose Tickers", ["AAPL", "MSFT", "JNJ", "NVDA", "GOOGL", "XOM", "V"], default=["AAPL", "MSFT", "JNJ"])
    start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.02, 0.005)
    apply_esg = st.sidebar.checkbox("Apply ESG Filter", value=True)
    apply_clustering = st.sidebar.checkbox("Apply Clustering for Diversification", value=True)

    # Data validation
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    if not tickers:
        st.warning("Please select at least one stock ticker.")
        return

    # ESG Filtering
    if apply_esg:
        tickers = esg_filter(tickers)
        st.info(f"Tickers after ESG filter: {tickers}")

    # Download data
    price_data = download_price_data(tickers, start_date, end_date)
    if price_data.empty:
        st.warning("No data returned. Try adjusting tickers or dates.")
        return

    st.subheader(" Downloaded Price Data")
    st.dataframe(price_data.tail())

    # Calculate returns
    returns = calculate_returns(price_data)
    if returns.empty:
        st.warning("Not enough data to calculate returns.")
        return

    # Clustering
    if apply_clustering:
        clusters = cluster_stocks(returns)
        st.subheader(" Stock Clusters")
        st.write(clusters)

    # Optimize
    weights = optimize_portfolio(returns, risk_free_rate)
    st.subheader(" Optimized Portfolio Weights")
    weight_df = pd.DataFrame(weights, index=returns.columns, columns=["Weight"])
    st.dataframe(weight_df)

    # Plot Efficient Frontier
    st.subheader("Efficient Frontier")
    plot_efficient_frontier(returns, risk_free_rate)

# Run the app
run_app()
