import pandas as pd


class TradingStrategy:
    def __init__(self, data, strategy):
        self.data = data.copy()
        self.data['signal'] = 'HOLD'  # Initialize all signals to HOLD
        self.strategy = strategy.upper()
        self.apply_strategy()

    def apply_strategy(self):
        if self.strategy == 'MACD':
            self.apply_macd_strategy()
        elif self.strategy == 'ML':
            self.apply_ml_strategy()
        elif self.strategy == 'COMBINED':
            self.apply_combined_strategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def apply_ml_strategy(self):
        if 'action_label' not in self.data.columns:
            raise ValueError("Expected 'action_label' column in data")

        self.data['ml_signal'] = 'HOLD'
        self.data.loc[self.data['action_label'] == 2, 'ml_signal'] = 'BUY'
        self.data.loc[self.data['action_label'] == 0, 'ml_signal'] = 'SELL'

        self.data['signal'] = self.data['ml_signal']

    def apply_macd_strategy(self):
        grouped = self.data.groupby('name', group_keys=False)

        def macd_signals(stock_df):
            stock_df['macd_signal_flag'] = 'HOLD'
            for i in range(1, len(stock_df)):
                if stock_df['macd'].iloc[i] > stock_df['macd_signal'].iloc[i] and \
                        stock_df['macd'].iloc[i - 1] <= stock_df['macd_signal'].iloc[i - 1]:
                    stock_df.loc[stock_df.index[i], 'macd_signal_flag'] = 'BUY'
                elif stock_df['macd'].iloc[i] < stock_df['macd_signal'].iloc[i] and \
                        stock_df['macd'].iloc[i - 1] >= stock_df['macd_signal'].iloc[i - 1]:
                    stock_df.loc[stock_df.index[i], 'macd_signal_flag'] = 'SELL'
            return stock_df

        self.data = grouped.apply(macd_signals)
        self.data['signal'] = self.data['macd_signal_flag']

    def apply_combined_strategy(self):
        self.apply_ml_strategy()
        self.apply_macd_strategy()

        def combine_signals(row):
            ml = row.get('ml_signal', 'HOLD')
            macd = row.get('macd_signal_flag', 'HOLD')

            if ml == 'BUY' and macd == 'BUY':
                return 'BUY'
            elif ml == 'SELL' or macd == 'SELL':
                return 'SELL'
            else:
                return 'HOLD'

        self.data['signal'] = self.data.apply(combine_signals, axis=1)
