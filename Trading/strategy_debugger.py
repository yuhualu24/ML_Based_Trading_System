import pandas as pd
import matplotlib.pyplot as plt


class StrategyDebugger:
    def __init__(self, strategy_data, trader, signal_col='signal', stock_col='name', date_col='time_key'):
        self.data = strategy_data.copy()
        self.trader = trader
        self.signal_col = signal_col
        self.stock_col = stock_col
        self.date_col = date_col

    def identify_loss_periods(self, threshold_pct_drop=5.0, window=5):

        issues = []
        grouped = self.data.groupby(self.stock_col)

        for stock, group in grouped:
            group = group.sort_values(by=self.date_col).reset_index(drop=True)
            group['pct_change'] = group['close'].pct_change(periods=window) * 100

            for i in range(len(group) - window):
                start_row = group.iloc[i]
                end_row = group.iloc[i + window]

                price_drop = (end_row['close'] - start_row['close']) / start_row['close'] * 100

                if price_drop <= -threshold_pct_drop:
                    window_df = group.iloc[i:i + window + 1]
                    signals = window_df[self.signal_col].values
                    if 'SELL' not in signals:
                        issues.append({
                            'stock': stock,
                            'start_date': start_row[self.date_col],
                            'end_date': end_row[self.date_col],
                            'drop_pct': round(price_drop, 2)
                        })

        return pd.DataFrame(issues)

    def visualize_issue_segment(self, stock, start_date, end_date=None, save_path=None):
        df = self.data[self.data[self.stock_col] == stock].copy()
        df['time_key'] = pd.to_datetime(df['time_key'])
        df = df[df[self.date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[self.date_col] <= pd.to_datetime(end_date)]

        if df.empty:
            print("No data available for selected stock and date range.")
            return

        plt.figure(figsize=(14, 6))
        plt.plot(df[self.date_col], df['close'], label='Close Price', color='black')

        buy_signals = df[df[self.signal_col] == 'BUY']
        plt.scatter(buy_signals[self.date_col], buy_signals['close'], marker='^', color='green', label='BUY', zorder=5)

        sell_signals = df[df[self.signal_col] == 'SELL']
        plt.scatter(sell_signals[self.date_col], sell_signals['close'], marker='v', color='red', label='SELL', zorder=5)

        plt.title(f'{stock} Price & Signals [{start_date} to {end_date or "end"}]')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=300)

    def summarize_issues(self, threshold_pct_drop=5.0, window=5):
        issues_df = self.identify_loss_periods(threshold_pct_drop=threshold_pct_drop, window=window)
        print(
            f"\n Found {len(issues_df)} issue segments where price dropped > {threshold_pct_drop}% without SELL signal\n")
        return issues_df
