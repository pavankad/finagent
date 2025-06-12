import pandas as pd
import numpy as np
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl import config
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from pathlib import Path
import pdb

MODEL_PATH = Path("/Users/pavan/vscode/finagent/agents/models/trained_ppo.zip")

# Environment creation
def create_env(df):
    pdb.set_trace()
    env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": 5,
    "stock_dim": 5,
    "tech_indicator_list": config.INDICATORS,
    "action_space": 5,
    "reward_scaling": 1e-4
}
    env = StockPortfolioEnv(df=df, **env_kwargs)
    return env


# Load existing agent
def load_trading_agent():
    model = PPO.load(MODEL_PATH)
    return model

# Simulate live environment
def make_trading_decision(latest_data):
    pdb.set_trace()
    env = create_env(latest_data)
    model = load_trading_agent()
    pdb.set_trace()
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model, environment = env)
    return df_daily_return, df_actions  # Maps to asset allocation decision

# Optional: Evaluate agent performance
def evaluate_trading_agent(df):
    env = create_env(add_technical_indicators(df))
    model = load_trading_agent()
    obs = env.reset()
    rewards = []
    for _ in range(len(df)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    return np.sum(rewards)


if __name__ == "__main__":
    df = pd.read_csv('/Users/pavan/vscode/finagent/data/processed/ticker_data')  # Load your stock data
    df = df.rename(columns={'timestamp': 'date'})
    pdb.set_trace()
    #add technical indicators
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

    df = fe.preprocess_data(df)
    
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=60
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)


    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    pdb.set_trace()
    trade = data_split(df, '2021-08-02', '2021-08-04')
    print(df)
    df_daily_return, df_actions = make_trading_decision(trade)
    print("Daily Returns:\n", df_daily_return)
    print("Actions:\n", df_actions)
    