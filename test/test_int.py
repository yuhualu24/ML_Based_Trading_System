#!/usr/bin/env python3
"""
ML Trading System Integration Test

This test validates the complete ML trading pipeline from data loading through
strategy execution and performance evaluation. It tests the integration of:

- Feature engineering and labeling
- ML model training (Random Forest and XGBoost)
- Trading strategy implementation (MACD, ML, Combined)
- Strategy debugging and visualization
- Performance metrics calculation

Requirements:
- sample_candlestick_data.csv in the current directory
- All Trading/, Models/, and Feature/ modules accessible

Quick Start:
    python test_integration.py

The test will:
1. Load sample stock data
2. Engineer features and create labels
3. Train classification models
4. Execute trading strategies
5. Generate debug visualizations
6. Save performance metrics
"""

import unittest
import sys
import os
import time
import datetime
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from Trading.trader import Trader
from Trading.trading_strategy import TradingStrategy
from Trading.strategy_debugger import StrategyDebugger
from Models.classification import ClassificationModel
from Feature.feature import FeatureEngineering, LabelingHelper


class TestMLTradingIntegration(unittest.TestCase):
    """
    Integration test for the complete ML trading pipeline

    Tests the end-to-end workflow from raw stock data to trading performance
    evaluation, ensuring all components work together correctly.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment and validate data availability"""
        print("\n" + "=" * 60)
        print("ML TRADING SYSTEM INTEGRATION TEST")
        print("=" * 60)

        # Check for required data file
        cls.data_file = '/Users/yuhualu24/Desktop/Trading_System_Project/test/100_stocks_data_by_Apr_2025.csv'
        if not os.path.exists(cls.data_file):
            raise FileNotFoundError(
                f"Required data file not found: {cls.data_file}"
            )

        print(f"Data file found: {cls.data_file}")

        cls.output_dir = os.path.join(os.getcwd(), "test_results")
        os.makedirs(cls.output_dir, exist_ok=True)
        print(f"Output directory: {cls.output_dir}")

    def test_complete_trading_pipeline(self):
        """
        Test the complete ML trading pipeline integration

        This single comprehensive test validates the entire workflow to ensure
        all components integrate properly without testing individual units.
        """
        print("\n--- TESTING COMPLETE TRADING PIPELINE ---")

        pipeline_start_time = time.time()

        # Step 1: Load and validate data
        print("\n1. Loading stock data...")
        try:
            raw_data = pd.read_csv(self.data_file)
            raw_data.columns = [
                'code', 'name', 'time_key', 'open', 'close', 'high', 'low',
                'pe_ratio', 'turnover_rate', 'volume', 'turnover', 'change_rate', 'last_close'
            ]
            print(f"Loaded {len(raw_data):,} records for {raw_data['code'].nunique()} stocks")
            print(f"Date range: {raw_data['time_key'].min()} to {raw_data['time_key'].max()}")
        except Exception as e:
            self.fail(f"Failed to load data: {e}")

        # Step 2: Feature engineering
        print("\n2. Engineering features...")
        try:
            # MACD parameters
            macd_fast, macd_slow, macd_signal = 6, 13, 4

            feature_engineer = FeatureEngineering(raw_data, macd_fast, macd_slow, macd_signal)
            labeler = LabelingHelper(feature_engineer.stock_data)

            print(f"Features engineered successfully")
            print(f"Training data shape: {labeler.classification_training_data.shape}")
            print(f"Original data shape: {labeler.original_data.shape}")
        except Exception as e:
            self.fail(f"Failed in feature engineering: {e}")

        # Step 3: Train ML models
        print("\n3. Training ML models...")
        try:
            # Train Random Forest classifier
            rf_classifier = ClassificationModel(
                labeler.classification_training_data,
                labeler.original_data,
                model_type='random_forest'
            )
            rf_classifier.train()
            print(
                f"Random Forest - Train: {rf_classifier.train_accuracy:.3f}, Test: {rf_classifier.test_accuracy:.3f}")

            # Train XGBoost classifier
            xgb_classifier = ClassificationModel(
                labeler.classification_training_data,
                labeler.original_data,
                model_type='xgboost'
            )
            xgb_classifier.train()
            print(f"XGBoost - Train: {xgb_classifier.train_accuracy:.3f}, Test: {xgb_classifier.test_accuracy:.3f}")

        except Exception as e:
            self.fail(f"Failed in model training: {e}")

        # Step 4: Create trading strategies
        print("\n4. Creating trading strategies...")
        try:
            # Use RF classifier for primary strategies
            macd_strategy = TradingStrategy(rf_classifier.full_data, 'MACD')
            ml_strategy = TradingStrategy(rf_classifier.full_data, 'ML')
            combined_strategy = TradingStrategy(rf_classifier.full_data, 'COMBINED')

            print("MACD strategy created")
            print("ML strategy created")
            print("Combined strategy created")
        except Exception as e:
            self.fail(f"Failed in strategy creation: {e}")

        # Step 5: Execute trading strategies
        print("\n5. Executing trading strategies...")
        try:
            macd_trader = Trader(macd_strategy.data)
            ml_trader = Trader(ml_strategy.data)
            combined_trader = Trader(combined_strategy.data)

            print("All trading strategies executed successfully")
        except Exception as e:
            self.fail(f"Failed in strategy execution: {e}")

        # Step 6: Generate debug visualizations
        print("\n6. Generating debug visualizations...")
        try:
            # Select random stock for visualization
            available_stocks = macd_trader.data['name'].unique()
            test_stock = random.choice(available_stocks)
            print(f"Using stock for visualization: {test_stock}")

            # Generate timestamp for unique filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Debug date range
            debug_start_date = '2020-01-01'
            debug_end_date = '2021-01-01'

            # Create debuggers and generate plots
            strategies_and_names = [
                (macd_strategy.data, macd_trader, 'macd'),
                (ml_strategy.data, ml_trader, 'ml'),
                (combined_strategy.data, combined_trader, 'combined')
            ]

            for strategy_data, trader, strategy_name in strategies_and_names:
                debugger = StrategyDebugger(strategy_data, trader)
                plot_path = os.path.join(self.output_dir, f'{strategy_name}_strategy_{timestamp}.png')

                debugger.visualize_issue_segment(test_stock, debug_start_date, debug_end_date, plot_path)
                print(f"{strategy_name.upper()} strategy visualization saved")

        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
            print("Continuing with performance evaluation...")

        # Step 7: Calculate and save performance metrics
        print("\n7. Calculating performance metrics...")
        try:
            pipeline_end_time = time.time()
            pipeline_runtime = int(pipeline_end_time - pipeline_start_time)

            # Prepare performance data
            traders_info = [
                ("MACD Strategy", macd_trader),
                ("ML Strategy", ml_trader),
                ("Combined Strategy", combined_trader)
            ]

            performance_data = []
            for strategy_name, trader in traders_info:
                final_cash = trader.cash
                final_portfolio_value = trader.portfolio_value[-1][1]
                daily_sharpe = trader.sharpe_ratio
                annualized_sharpe = daily_sharpe * (252 ** 0.5)

                strategy_performance = {
                    'strategy': strategy_name,
                    'final_cash': final_cash,
                    'final_portfolio_value': final_portfolio_value,
                    'daily_sharpe_ratio': daily_sharpe,
                    'annualized_sharpe_ratio': annualized_sharpe
                }
                performance_data.append(strategy_performance)

                print(f"{strategy_name}: Final Value ${final_portfolio_value:,.2f}, Sharpe {annualized_sharpe:.3f}")

            # Save detailed results to log file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.output_dir, f'integration_test_results_{timestamp}.txt')

            with open(log_file, 'w') as f:
                f.write("ML Trading System Integration Test Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Timestamp: {timestamp}\n")
                f.write(f"Total Runtime: {pipeline_runtime} seconds\n")
                f.write(f"Number of Stocks: {raw_data['code'].nunique()}\n")
                f.write(f"Stock Symbols: {', '.join(raw_data['code'].unique()[:10])}...\n")  # Show first 10
                f.write(f"MACD Parameters (fast, slow, signal): {macd_fast}, {macd_slow}, {macd_signal}\n\n")

                f.write("Model Performance:\n")
                f.write(f"Random Forest - Train Accuracy: {rf_classifier.train_accuracy:.5f}\n")
                f.write(f"Random Forest - Test Accuracy: {rf_classifier.test_accuracy:.5f}\n")
                f.write(f"XGBoost - Train Accuracy: {xgb_classifier.train_accuracy:.5f}\n")
                f.write(f"XGBoost - Test Accuracy: {xgb_classifier.test_accuracy:.5f}\n\n")

                f.write("Trading Strategy Performance:\n")
                for perf in performance_data:
                    f.write(f"\n{perf['strategy']}:\n")
                    f.write(f"  Final Cash: ${perf['final_cash']:,.2f}\n")
                    f.write(f"  Final Portfolio Value: ${perf['final_portfolio_value']:,.2f}\n")
                    f.write(f"  Daily Sharpe Ratio: {perf['daily_sharpe_ratio']:.5f}\n")
                    f.write(f"  Annualized Sharpe Ratio: {perf['annualized_sharpe_ratio']:.5f}\n")

            print(f"Detailed results saved to: {log_file}")

        except Exception as e:
            self.fail(f"Failed in performance calculation: {e}")

        print(f"\nIntegration test completed successfully in {pipeline_runtime} seconds!")
        print(f"Results saved to: {self.output_dir}")

    def test_data_integrity(self):
        """
        Quick test to validate data integrity and basic pipeline components

        This lightweight test ensures the data and core components are working
        without running the full pipeline.
        """
        print("\n--- TESTING DATA INTEGRITY ---")

        # Load and validate data structure
        raw_data = pd.read_csv(self.data_file)
        raw_data.columns = [
            'code', 'name', 'time_key', 'open', 'close', 'high', 'low',
            'pe_ratio', 'turnover_rate', 'volume', 'turnover', 'change_rate', 'last_close'
        ]

        # Basic data validation
        self.assertFalse(raw_data.empty, "Data should not be empty")
        self.assertGreater(raw_data['code'].nunique(), 0, "Should have at least one stock")

        # Check required columns exist
        required_columns = ['code', 'name', 'time_key', 'open', 'close', 'high', 'low', 'volume']
        for col in required_columns:
            self.assertIn(col, raw_data.columns, f"Missing required column: {col}")

        # Check data types and ranges
        price_columns = ['open', 'close', 'high', 'low', 'volume']
        for col in price_columns:
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
        self.assertTrue((raw_data['high'] >= raw_data['low']).all(), "High should be >= Low")
        self.assertTrue((raw_data['volume'] >= 0).all(), "Volume should be non-negative")

        print("Data integrity validation passed")

    @classmethod
    def tearDownClass(cls):
        """Clean up and provide final summary"""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST COMPLETED")
        print("=" * 60)
        print(f"Results and visualizations saved to: {cls.output_dir}")
        print("\nNext steps:")
        print("1. Review the performance metrics in the log file")
        print("2. Examine the debug visualizations")
        print("3. Optimize strategy parameters based on results")
        print("4. Test with different time periods or stocks")
        print("=" * 60)


if __name__ == "__main__":
    print("Starting ML Trading System Integration Test...")
    print("Use this test to validate the complete trading pipeline")
    unittest.main(verbosity=2)