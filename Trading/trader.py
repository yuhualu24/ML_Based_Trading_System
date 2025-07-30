import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class Trader:
    def __init__(
        self,
        data,
        initial_cash=1000000,
        transaction_cost=0.001,
        signal_col='signal',
        position_sizing='fixed_shares',  # 'fixed_shares' or 'fixed_dollar'
        shares_per_trade=10,
        dollar_per_trade=10000,
        max_position_scaling=3,
        scaling_threshold_pct=2.0,
        sell_on_consecutive_down_days=5,
        stock_col='name',
        date_col='time_key'
    ):
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.transaction_cost = transaction_cost

        self.signal_col = signal_col
        self.stock_col = stock_col
        self.date_col = date_col

        self.position_sizing = position_sizing
        self.shares_per_trade = shares_per_trade
        self.dollar_per_trade = dollar_per_trade

        self.max_position_scaling = max_position_scaling
        self.scaling_threshold_pct = scaling_threshold_pct  # Price must increase by this percent to scale further

        self.sell_on_consecutive_down_days = sell_on_consecutive_down_days

        self.positions = defaultdict(int)
        self.last_buy_price = defaultdict(lambda: None)
        self.scaling_steps = defaultdict(int)
        self.trade_log = []
        self.portfolio_value = []
        self.latest_prices = {}

        self.execute_trades()
        self.sharpe_ratio = self.calculate_sharpe_ratio()

    def execute_trades(self):
        grouped = self.data.groupby(self.stock_col)

        for stock, group in grouped:
            consecutive_down_counter = 0

            for i, (idx, row) in enumerate(group.iterrows()):
                price = row['close']
                signal = row.get(self.signal_col, 'HOLD')
                date = row[self.date_col]
                self.latest_prices[stock] = price

                # Track consecutive down days
                if i > 0 and group['close'].iloc[i] < group['close'].iloc[i - 1]:
                    consecutive_down_counter += 1
                else:
                    consecutive_down_counter = 0

                # === BUY Logic with Position Scaling ===
                if signal == 'BUY':
                    allow_additional_buy = (
                        self.scaling_steps[stock] < self.max_position_scaling and
                        self.last_buy_price[stock] is not None and
                        price >= self.last_buy_price[stock] * (1 + self.scaling_threshold_pct / 100)
                    )
                    initial_buy = self.positions[stock] == 0

                    if initial_buy or allow_additional_buy:
                        shares_to_buy = self._calculate_shares(price)
                        cost = shares_to_buy * price * (1 + self.transaction_cost)

                        if shares_to_buy > 0 and self.cash >= cost:
                            self.positions[stock] += shares_to_buy
                            self.cash -= cost
                            self.last_buy_price[stock] = price
                            self.scaling_steps[stock] += 1
                            self.trade_log.append((date, stock, 'BUY', shares_to_buy, price, self.cash))

                # === SELL Logic ===
                elif signal == 'SELL' and self.positions[stock] > 0:
                    shares_to_sell = self.positions[stock]
                    revenue = shares_to_sell * price * (1 - self.transaction_cost)
                    self.cash += revenue
                    self.positions[stock] = 0
                    self.scaling_steps[stock] = 0
                    self.last_buy_price[stock] = None
                    self.trade_log.append((date, stock, 'SELL', shares_to_sell, price, self.cash))

                # === AUTO SELL on Too Many Down Days ===
                elif consecutive_down_counter >= self.sell_on_consecutive_down_days and self.positions[stock] > 0:
                    shares_to_sell = self.positions[stock]
                    revenue = shares_to_sell * price * (1 - self.transaction_cost)
                    self.cash += revenue
                    self.positions[stock] = 0
                    self.scaling_steps[stock] = 0
                    self.last_buy_price[stock] = None
                    self.trade_log.append((date, stock, 'AUTO SELL (Too Many Down Days)', shares_to_sell, price, self.cash))
                    consecutive_down_counter = 0

                # === Record Portfolio Value ===
                total_position_value = sum(
                    self.positions[s] * self.latest_prices.get(s, 0)
                    for s in self.positions
                )
                portfolio_value = self.cash + total_position_value
                self.portfolio_value.append((date, portfolio_value))

    def _calculate_shares(self, price):
        if self.position_sizing == 'fixed_shares':
            return self.shares_per_trade
        elif self.position_sizing == 'fixed_dollar':
            return int(self.dollar_per_trade // price)
        else:
            raise ValueError("Invalid position sizing method")

    def get_trade_log(self):
        return pd.DataFrame(self.trade_log, columns=['Date', 'Stock', 'Action', 'Shares', 'Price', 'CashAfter'])

    def get_portfolio_value(self):
        return pd.DataFrame(self.portfolio_value, columns=['Date', 'PortfolioValue'])

    def summary(self):
        final_value = self.portfolio_value[-1][1] if self.portfolio_value else self.cash
        return {
            'Final Portfolio Value': final_value,
            'Cash Remaining': self.cash,
            'Final Positions': dict(self.positions),
            'Total Trades': len(self.trade_log)
        }

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        portfolio_df = self.get_portfolio_value()
        portfolio_df['PortfolioReturn'] = portfolio_df['PortfolioValue'].pct_change().fillna(0)

        excess_returns = portfolio_df['PortfolioReturn'] - risk_free_rate
        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        sharpe_ratio = mean_return / std_return if std_return != 0 else float('nan')
        return sharpe_ratio

    def plot_portfolio_value(self):
        df = self.get_portfolio_value()
        plt.figure(figsize=(14, 6))
        plt.plot(pd.to_datetime(df['Date']), df['PortfolioValue'], label='Portfolio Value', color='blue')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_top_k_positions(trade_log, full_data, top_k=10, date_col='time_key'):
        """
        Visualize share positions over time for the top K most traded stocks.

        :param trade_log: DataFrame from Trader.get_trade_log()
        :param full_data: Full strategy DataFrame containing all dates per stock
        :param top_k: Number of top stocks (by trade count) to visualize
        :param date_col: Column name representing dates (e.g. 'time_key')
        """
        # Get top K most frequently traded stocks
        top_stocks = trade_log['Stock'].value_counts().nlargest(top_k).index.tolist()

        # Create a full index of all dates for each top stock
        date_index = (
            full_data[full_data['name'].isin(top_stocks)]
            .groupby('name')[date_col]
            .apply(list)
            .explode()
            .drop_duplicates()
            .sort_values()
        )

        # Initialize share matrix
        positions = {stock: pd.Series(0, index=date_index.unique()) for stock in top_stocks}

        # Sort trade log chronologically
        trade_log_sorted = trade_log[trade_log['Stock'].isin(top_stocks)].sort_values(by='Date')

        for _, row in trade_log_sorted.iterrows():
            stock = row['Stock']
            date = row['Date']
            action = row['Action']
            shares = row['Shares']

            if action.startswith('BUY'):
                positions[stock].loc[positions[stock].index >= date] += shares
            elif action.startswith('SELL'):
                positions[stock].loc[positions[stock].index >= date] -= shares

        # Combine into DataFrame
        pos_df = pd.DataFrame(positions)

        # Plot
        plt.figure(figsize=(16, 6))
        for stock in pos_df.columns:
            plt.plot(pos_df.index, pos_df[stock], label=stock)
        plt.title(f"Top {top_k} Stock Holdings Over Time")
        plt.xlabel("Date")
        plt.ylabel("Shares Held")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
