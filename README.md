# RL-Based Trading Bot

A reinforcement learning-based stock trading system that uses a **Deep Q-Network (DQN)** to learn trading decisions — **BUY, SELL, or HOLD** — from historical stock market data.

The current implementation uses **Apple (AAPL)** historical data and is implemented in Python using PyTorch, Pandas, NumPy, and yfinance.

## Overview

This project demonstrates how reinforcement learning can be applied to a basic algorithmic trading problem.

The system:

1. Downloads historical AAPL stock data.
2. Performs feature engineering using moving averages and returns.
3. Creates a custom trading environment.
4. Defines BUY, SELL, and HOLD actions.
5. Builds a Deep Q-Network (DQN).
6. Trains the agent using experience replay and epsilon-greedy exploration.
7. Tests the trained agent on the available historical data.
8. Reports the final portfolio balance and profit.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- yfinance
- Jupyter Notebook

## Dataset

Historical stock market data is downloaded using the `yfinance` library.

### Stock

**Ticker:** AAPL (Apple Inc.)

### Period

- Start: `2020-01-01`
- End: `2025-02-14`

The downloaded data contains:

- Open
- High
- Low
- Close
- Volume

## Feature Engineering

Three additional features are calculated from the historical closing price:

### 5-Day Simple Moving Average

```python
SMA_5 = Close.rolling(window=5).mean()
```

### 20-Day Simple Moving Average

```python
SMA_20 = Close.rolling(window=20).mean()
```

### Daily Returns

```python
Returns = Close.pct_change()
```

Rows containing missing values are removed before training.

## State Representation

The agent observes four features at every timestep:

```text
[Close, SMA_5, SMA_20, Returns]
```

Therefore, the state size is **4**.

## Action Space

The trading agent has three possible actions:

| Action | Decision |
|---|---|
| 0 | HOLD |
| 1 | BUY |
| 2 | SELL |

## Trading Environment

A custom `TradingEnvironment` class is used to simulate trading.

### Initial Capital

```text
$10,000
```

The environment maintains:

- Cash balance
- Stock holdings
- Current market index

When a BUY action is selected, the agent purchases as many shares as possible with the available balance.

When a SELL action is selected, all currently held shares are sold at the current price.

The reward is calculated at the end of the trading episode based on the difference between the final balance and the initial balance.

## Deep Q-Network

The project uses a fully connected neural network as the Q-network.

### Architecture

```text
Input Layer
    ↓
4 Features
    ↓
Fully Connected Layer
64 neurons
    ↓
Fully Connected Layer
64 neurons
    ↓
Output Layer
3 Actions
```

The three output values correspond to the Q-values for:

```text
HOLD
BUY
SELL
```

ReLU activation is used between the fully connected layers.

## DQN Agent

The agent uses:

- Experience replay
- Epsilon-greedy exploration
- Adam optimizer
- Mean Squared Error loss

### Main Parameters

```text
Replay Memory: 2000 experiences
Discount Factor (gamma): 0.95
Initial Epsilon: 1.0
Minimum Epsilon: 0.01
Epsilon Decay: 0.995
Learning Rate: 0.001
Batch Size: 32
Training Episodes: 500
```

## Training

The agent is trained for **500 episodes**.

During training, the agent:

1. Resets the trading environment.
2. Observes the current state.
3. Selects an action using epsilon-greedy exploration.
4. Executes the action.
5. Receives the next state and reward.
6. Stores the experience in replay memory.
7. Uses sampled experiences to update the neural network.
8. Gradually reduces exploration through epsilon decay.

## Results

The notebook contains training output for 500 episodes and evaluates the trained agent by simulating a trading session.

One recorded test run produced:

```text
Initial Balance: $10,000
Final Balance: $12,075.09
Total Profit: $2,075.09
```

Another recorded test run produced:

```text
Initial Balance: $10,000
Final Balance: $10,238.46
Total Profit: $238.46
```

The difference between these results reflects the behavior of the trained agent during the respective test executions.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Srinu1802/RL-based-Trading-bot.git
cd RL-based-Trading-bot
```

### 2. Install dependencies

```bash
pip install yfinance pandas numpy torch
```

### 3. Open the notebook

Open:

```text
main.ipynb
```

using Jupyter Notebook, JupyterLab, or VS Code.

### 4. Run the notebook

Execute the cells sequentially to:

- Download the stock data
- Generate features
- Create the trading environment
- Initialize the DQN agent
- Train the agent
- Test the trained model

## Project Structure

```text
RL-based-Trading-bot/
│
├── main.ipynb
└── README.md
```

## Important Note

This project is an **experimental reinforcement learning implementation for educational purposes**. The reported results are based on historical data and the trading environment implemented in the notebook.

The model should not be considered a production-ready financial trading system or a guarantee of future market performance.

## Future Improvements

Possible improvements include:

- Train on multiple stocks instead of only AAPL.
- Separate training and testing datasets.
- Add transaction costs and brokerage fees.
- Add stop-loss and risk-management mechanisms.
- Include additional technical indicators.
- Improve the reward function.
- Use a target network for more stable DQN training.
- Implement Double DQN.
- Implement Dueling DQN.
- Perform proper backtesting on unseen data.
- Compare the RL strategy against a buy-and-hold baseline.
- Add portfolio-level trading across multiple assets.
- Visualize trading signals and portfolio performance.

## License

This project is intended for educational and experimental use.
