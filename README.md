# ML-Based Stock Trading System

This project explores a machine learning driven trading system that uses classification and transformer models and 
technical indicators to generate BUY/HOLD/SELL signals for stock trading. Results suggest that ML-based models capture 
market patterns much more effectively than traditional MACD-based approaches.


## 🎯 Overview

This system combines:
- **ML Models**: Random Forest, XGBoost, Transformer-based neural networks with attention mechanisms
- **Technical Indicators**: MACD, RSI, moving averages, etc.
- **Multiple Strategies**: Pure ML, traditional MACD, and hybrid approaches
- **Comprehensive Testing**: Integration tests and performance evaluation
- **Visualization Tools**: Strategy debugging and performance analysis

The system processes **daily stock candlestick data**, engineers features from historical data, trains ML models, 
and generates **next-day trading signals** (BUY/HOLD/SELL) with detailed performance tracking.

## 🚀 Key Features

- **Daily Trading Signals**: Generates next-day BUY/HOLD/SELL predictions
- **Transformer Model**: Attention-based neural network for sequence prediction
- **Feature Engineering**: 25+ technical indicators and price-based features
- **Multi-Strategy Trading**: MACD, ML-based, and combined approaches
- **Risk Management**: Sharpe ratio optimization and portfolio tracking
- **Performance Visualization**: In-depth charts for examining stock-level trading decisions and portfolio evolution
- **Comprehensive Testing**: Full integration test suite

## 📊 Required Data Format

### Input Data Columns

Your stock data CSV file must contain the following columns:

| Column | Name | Type | Description | Example |
|--------|------|------|-------------|--------|
| 1 | `code` | string | Stock symbol/ticker | `US.AAPL` |
| 2 | `name` | string | Company name | `Apple Inc` |
| 3 | `time_key` | datetime | Date and time | `2023-04-20 00:00:00` |
| 4 | `open` | float | Opening price | `150.25` |
| 5 | `close` | float | Closing price | `152.30` |
| 6 | `high` | float | Highest price | `153.45` |
| 7 | `low` | float | Lowest price | `149.80` |
| 8 | `pe_ratio` | float | Price-to-earnings ratio | `25.5` |
| 9 | `turnover_rate` | float | Daily turnover rate (%) | `0.03124` |
| 10 | `volume` | integer | Trading volume | `1827006` |
| 11 | `turnover` | float | Total turnover amount | `0.0` |
| 12 | `change_rate` | float | Daily price change (%) | `1.234922458357270` |
| 13 | `last_close` | float | Previous day's close | `150.75` |

### Example Data Rows
```csv
US.AAPL,Apple Inc,2023-04-20 00:00:00,150.25,152.30,153.45,149.80,25.5,0.03124,1827006,0.0,1.234922458357270,150.75
US.GOOGL,Alphabet Inc,2023-04-20 00:00:00,105.50,107.20,108.00,104.90,22.1,0.02156,987543,0.0,1.610283687943262,105.72
```
