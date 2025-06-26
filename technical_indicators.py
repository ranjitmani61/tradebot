"""
Technical indicators calculation module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            return ta.RSI(data['Close'].values, timeperiod=period)
        except Exception:
            # Fallback calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            macd, macd_signal, macd_histogram = ta.MACD(data['Close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd, index=data.index),
                'macd_signal': pd.Series(macd_signal, index=data.index),
                'macd_histogram': pd.Series(macd_histogram, index=data.index)
            }
        except Exception:
            # Fallback calculation
            ema_fast = data['Close'].ewm(span=fast).mean()
            ema_slow = data['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return {
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram
            }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            bb_upper, bb_middle, bb_lower = ta.BBANDS(data['Close'].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return {
                'bb_upper': pd.Series(bb_upper, index=data.index),
                'bb_middle': pd.Series(bb_middle, index=data.index),
                'bb_lower': pd.Series(bb_lower, index=data.index)
            }
        except Exception:
            # Fallback calculation
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            
            return {
                'bb_upper': sma + (std * std_dev),
                'bb_middle': sma,
                'bb_lower': sma - (std * std_dev)
            }
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            return ta.SMA(data['Close'].values, timeperiod=period)
        except Exception:
            return data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return ta.EMA(data['Close'].values, timeperiod=period)
        except Exception:
            return data['Close'].ewm(span=period).mean()
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            return ta.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=period)
        except Exception:
            # Fallback calculation
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            return true_range.rolling(period).mean()
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            slowk, slowd = ta.STOCH(data['High'].values, data['Low'].values, data['Close'].values, 
                                   fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {
                'stoch_k': pd.Series(slowk, index=data.index),
                'stoch_d': pd.Series(slowd, index=data.index)
            }
        except Exception:
            # Fallback calculation
            lowest_low = data['Low'].rolling(window=k_period).min()
            highest_high = data['High'].rolling(window=k_period).max()
            
            k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'stoch_k': k_percent,
                'stoch_d': d_percent
            }
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            return ta.WILLR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=period)
        except Exception:
            # Fallback calculation
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            
            wr = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
            return wr
    
    def calculate_volume_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Simple Moving Average"""
        return data['Volume'].rolling(window=period).mean()
    
    def calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Ratio (current volume / average volume)"""
        volume_sma = self.calculate_volume_sma(data, period)
        return data['Volume'] / volume_sma
    
    def calculate_price_change(self, data: pd.DataFrame, periods: int = 1) -> pd.Series:
        """Calculate price change percentage"""
        return data['Close'].pct_change(periods) * 100
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Calculate basic support and resistance levels"""
        try:
            if len(data) < window:
                return {'support': 0, 'resistance': 0}
            
            # Use recent data for S&R calculation
            recent_data = data.tail(window)
            
            # Simple S&R based on min/max
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            
            return {
                'support': float(support),
                'resistance': float(resistance)
            }
            
        except Exception:
            return {'support': 0, 'resistance': 0}
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Pivot Points"""
        try:
            if len(data) < 2:
                return {}
            
            # Use previous day's data
            high = data['High'].iloc[-2]
            low = data['Low'].iloc[-2]
            close = data['Close'].iloc[-2]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': float(pivot),
                'r1': float(r1),
                'r2': float(r2),
                'r3': float(r3),
                's1': float(s1),
                's2': float(s2),
                's3': float(s3)
            }
            
        except Exception:
            return {}
    
    def calculate_fibonacci_retracements(self, data: pd.DataFrame, period: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if len(data) < period:
                return {}
            
            recent_data = data.tail(period)
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            
            diff = high - low
            
            levels = {
                '0%': float(high),
                '23.6%': float(high - 0.236 * diff),
                '38.2%': float(high - 0.382 * diff),
                '50%': float(high - 0.5 * diff),
                '61.8%': float(high - 0.618 * diff),
                '78.6%': float(high - 0.786 * diff),
                '100%': float(low)
            }
            
            return levels
            
        except Exception:
            return {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        try:
            if data.empty or len(data) < 20:
                return {}
            
            indicators = {}
            
            # Basic indicators
            indicators['rsi'] = float(self.calculate_rsi(data).iloc[-1]) if len(self.calculate_rsi(data).dropna()) > 0 else 50
            
            # MACD
            macd_data = self.calculate_macd(data)
            indicators['macd'] = float(macd_data['macd'].iloc[-1]) if len(macd_data['macd'].dropna()) > 0 else 0
            indicators['macd_signal'] = float(macd_data['macd_signal'].iloc[-1]) if len(macd_data['macd_signal'].dropna()) > 0 else 0
            indicators['macd_histogram'] = float(macd_data['macd_histogram'].iloc[-1]) if len(macd_data['macd_histogram'].dropna()) > 0 else 0
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(data)
            indicators['bb_upper'] = float(bb_data['bb_upper'].iloc[-1]) if len(bb_data['bb_upper'].dropna()) > 0 else 0
            indicators['bb_middle'] = float(bb_data['bb_middle'].iloc[-1]) if len(bb_data['bb_middle'].dropna()) > 0 else 0
            indicators['bb_lower'] = float(bb_data['bb_lower'].iloc[-1]) if len(bb_data['bb_lower'].dropna()) > 0 else 0
            
            # Moving Averages
            indicators['sma_5'] = float(self.calculate_sma(data, 5).iloc[-1]) if len(self.calculate_sma(data, 5).dropna()) > 0 else 0
            indicators['sma_10'] = float(self.calculate_sma(data, 10).iloc[-1]) if len(self.calculate_sma(data, 10).dropna()) > 0 else 0
            indicators['sma_20'] = float(self.calculate_sma(data, 20).iloc[-1]) if len(self.calculate_sma(data, 20).dropna()) > 0 else 0
            indicators['ema_12'] = float(self.calculate_ema(data, 12).iloc[-1]) if len(self.calculate_ema(data, 12).dropna()) > 0 else 0
            indicators['ema_26'] = float(self.calculate_ema(data, 26).iloc[-1]) if len(self.calculate_ema(data, 26).dropna()) > 0 else 0
            
            # Other indicators
            indicators['atr'] = float(self.calculate_atr(data).iloc[-1]) if len(self.calculate_atr(data).dropna()) > 0 else 0
            indicators['volume_ratio'] = float(self.calculate_volume_ratio(data).iloc[-1]) if len(self.calculate_volume_ratio(data).dropna()) > 0 else 1
            indicators['price_change'] = float(self.calculate_price_change(data).iloc[-1]) if len(self.calculate_price_change(data).dropna()) > 0 else 0
            
            # Support and Resistance
            sr_levels = self.calculate_support_resistance(data)
            indicators.update(sr_levels)
            
            # Current price data
            indicators['current_price'] = float(data['Close'].iloc[-1])
            indicators['current_volume'] = float(data['Volume'].iloc[-1])
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
