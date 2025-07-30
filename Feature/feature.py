import pandas as pd
import talib


class FeatureEngineering:
    def __init__(self, stock_data, macd_fast, macd_slow, macd_signal):
        """
        Expects input data with standard candlestick columns: 'open', 'high', 'low', 'close', 'volume'
        Returns enhanced DataFrame with technical indicators as features.
        """
        self.stock_data = stock_data
        self.macd_fastperiod = macd_fast
        self.macd_slowperiod = macd_slow
        self.macd_signalperiod = macd_signal
        self.add_technical_indicators()

    def add_technical_indicators(self):
        def compute_indicators(group):
            group = group.sort_values('time_key').copy()
            group['macd'], group['macd_signal'], group['macd_hist'] = talib.MACD(group['close'], fastperiod=self.macd_fastperiod,
                                                                                 slowperiod=self.macd_slowperiod, signalperiod=self.macd_signalperiod)
            group['rsi_14'] = talib.RSI(group['close'], timeperiod=14)
            group['sma_5'] = talib.SMA(group['close'], timeperiod=5)
            group['sma_10'] = talib.SMA(group['close'], timeperiod=10)
            group['sma_30'] = talib.SMA(group['close'], timeperiod=30)
            group['ema_5'] = talib.EMA(group['close'], timeperiod=5)
            group['ema_10'] = talib.EMA(group['close'], timeperiod=10)
            group['ema_30'] = talib.EMA(group['close'], timeperiod=30)
            group['atr_14'] = talib.ATR(group['high'], group['low'], group['close'], timeperiod=14)
            group['mom_10'] = talib.MOM(group['close'], timeperiod=10)
            group['slowk'], group['slowd'] = talib.STOCH(group['high'], group['low'], group['close'],
                                                         fastk_period=14, slowk_period=3, slowk_matype=0,
                                                         slowd_period=3, slowd_matype=0)
            group['price_range'] = group['high'] - group['low']
            group['candle_body_size'] = abs(group['close'] - group['open'])
            group['upper_shadow'] = group['high'] - group[['close', 'open']].max(axis=1)
            group['lower_shadow'] = group[['close', 'open']].min(axis=1) - group['low']
            group['volatility_5d'] = group['close'].rolling(window=5).std()
            group['volatility_10d'] = group['close'].rolling(window=10).std()

            return group

        df = self.stock_data
        df = df.groupby('name', group_keys=False).apply(compute_indicators)
        self.stock_data = df


class LabelingHelper:
    def __init__(self, stock_data, stock_col='name', price_col='close'):
        """
        Helper class for assigning classification and regression labels per stock.
        :param data: DataFrame containing historical stock data
        :param stock_col: Column name for stock symbols (default 'name')
        :param price_col: Column name for closing price (default 'close')
        """
        self.original_data = stock_data
        self.stock_col = stock_col
        self.price_col = price_col
        # self.regression_training_data = self.assign_regression_target()
        self.classification_training_data = self.assign_classification_labels()


    def assign_classification_labels(self, bins=[-float('inf'), -1.5, 1.5, float('inf')], labels=[0, 1, 2]):
        """
        Assign multi-class action labels (Strong Sell to Strong Buy) based on next-day return.
        :param bins: Thresholds for binning returns
        :param labels: Corresponding label classes
        """
        self.original_data.sort_values(by=[self.stock_col, 'time_key'], inplace=True)
        self.original_data['next_day_return'] = self.original_data.groupby(self.stock_col)[self.price_col].shift(-1) / self.original_data[self.price_col] - 1
        self.original_data['next_day_return'] = self.original_data['next_day_return'] * 100
        self.original_data.dropna(subset='next_day_return', inplace=True)

        self.original_data['action_label'] = pd.cut(self.original_data['next_day_return'], bins=bins, labels=labels).astype(int)
        return self.original_data.copy().drop(['code','name','time_key', 'next_day_return'], axis=1)
