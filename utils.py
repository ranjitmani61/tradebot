"""
Utility functions for the Professional Intraday Trading Assistant
"""

import pandas as pd
import numpy as np
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

def load_configuration(config_file: str = 'config.py') -> Dict[str, Any]:
    """Load configuration from config file"""
    try:
        from config import (
            NSE_ALL_STOCKS, BSE_STOCKS, NIFTY_50_STOCKS, SECTORS,
            DEFAULT_WATCHLIST, TRADING_CONFIG, CHART_CONFIG,
            AI_CONFIG, TELEGRAM_CONFIG, ALERT_CONFIG,
            FIBONACCI_LEVELS, SUPPORT_RESISTANCE_CONFIG,
            MARKET_SESSIONS, DEFAULT_FEATURES, RISK_CONFIG,
            BACKTEST_CONFIG, PERFORMANCE_METRICS
        )
        
        return {
            'nse_stocks': NSE_ALL_STOCKS,
            'bse_stocks': BSE_STOCKS,
            'nifty_50': NIFTY_50_STOCKS,
            'sectors': SECTORS,
            'default_watchlist': DEFAULT_WATCHLIST,
            'trading_config': TRADING_CONFIG,
            'chart_config': CHART_CONFIG,
            'ai_config': AI_CONFIG,
            'telegram_config': TELEGRAM_CONFIG,
            'alert_config': ALERT_CONFIG,
            'fibonacci_levels': FIBONACCI_LEVELS,
            'support_resistance_config': SUPPORT_RESISTANCE_CONFIG,
            'market_sessions': MARKET_SESSIONS,
            'default_features': DEFAULT_FEATURES,
            'risk_config': RISK_CONFIG,
            'backtest_config': BACKTEST_CONFIG,
            'performance_metrics': PERFORMANCE_METRICS
        }
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return get_default_configuration()

def get_default_configuration() -> Dict[str, Any]:
    """Get default configuration when config file is not available"""
    return {
        'nse_stocks': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN'],
        'bse_stocks': ['RELIANCE', 'TCS', 'HDFCBANK'],
        'nifty_50': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN'],
        'sectors': ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG'],
        'default_watchlist': ['RELIANCE', 'TCS', 'HDFCBANK'],
        'trading_config': {
            'timeframes': ['1m', '5m', '15m', '30m', '1h'],
            'default_timeframe': '5m',
            'default_period': '1d'
        },
        'chart_config': {'height': 600, 'colors': {}},
        'ai_config': {'models': {}, 'features': []},
        'telegram_config': {'bot_token': '', 'chat_id': ''},
        'alert_config': {},
        'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
        'support_resistance_config': {},
        'market_sessions': {},
        'default_features': {},
        'risk_config': {},
        'backtest_config': {},
        'performance_metrics': []
    }

def validate_stock_symbol(symbol: str) -> bool:
    """Validate if stock symbol is valid"""
    try:
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation
        symbol = symbol.upper().strip()
        
        # Check length (typically 1-10 characters for Indian stocks)
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # Check for valid characters (alphanumeric only)
        if not symbol.replace('-', '').replace('&', '').isalnum():
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating symbol {symbol}: {str(e)}")
        return False

def format_currency(amount: float, currency: str = '₹') -> str:
    """Format currency amount for display"""
    try:
        if amount >= 10000000:  # 1 crore
            return f"{currency}{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"{currency}{amount/100000:.2f}L"
        elif amount >= 1000:  # 1 thousand
            return f"{currency}{amount/1000:.2f}K"
        else:
            return f"{currency}{amount:.2f}"
            
    except Exception as e:
        print(f"Error formatting currency: {str(e)}")
        return f"{currency}{amount}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value for display"""
    try:
        if value >= 0:
            return f"+{value:.{decimals}f}%"
        else:
            return f"{value:.{decimals}f}%"
            
    except Exception as e:
        print(f"Error formatting percentage: {str(e)}")
        return f"{value}%"

def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        if abs(number) >= 10000000:  # 1 crore
            return f"{number/10000000:.{decimals}f}Cr"
        elif abs(number) >= 100000:  # 1 lakh
            return f"{number/100000:.{decimals}f}L"
        elif abs(number) >= 1000:  # 1 thousand
            return f"{number/1000:.{decimals}f}K"
        else:
            return f"{number:.{decimals}f}"
            
    except Exception as e:
        print(f"Error formatting number: {str(e)}")
        return str(number)

def calculate_position_size(capital: float, risk_percentage: float, 
                          entry_price: float, stop_loss: float) -> Dict[str, Any]:
    """Calculate position size based on risk management"""
    try:
        risk_amount = capital * (risk_percentage / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {'error': 'Invalid stop loss level'}
        
        position_size = int(risk_amount / risk_per_share)
        position_value = position_size * entry_price
        actual_risk = position_size * risk_per_share
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'actual_risk': actual_risk,
            'risk_percentage': (actual_risk / capital) * 100,
            'risk_per_share': risk_per_share
        }
        
    except Exception as e:
        print(f"Error calculating position size: {str(e)}")
        return {'error': str(e)}

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, 
                               target_price: float) -> Dict[str, Any]:
    """Calculate risk-reward ratio for a trade"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        if risk <= 0:
            return {'error': 'Invalid risk calculation'}
        
        ratio = reward / risk
        
        return {
            'risk': risk,
            'reward': reward,
            'ratio': ratio,
            'quality': get_risk_reward_quality(ratio)
        }
        
    except Exception as e:
        print(f"Error calculating risk-reward ratio: {str(e)}")
        return {'error': str(e)}

def get_risk_reward_quality(ratio: float) -> str:
    """Assess quality of risk-reward ratio"""
    if ratio >= 3:
        return "Excellent"
    elif ratio >= 2:
        return "Good"
    elif ratio >= 1.5:
        return "Fair"
    elif ratio >= 1:
        return "Acceptable"
    else:
        return "Poor"

def export_data_to_csv(data: Union[pd.DataFrame, List[Dict]], filename: str) -> str:
    """Export data to CSV format"""
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if df.empty:
            return "No data to export"
        
        # Generate CSV string
        csv_string = df.to_csv(index=False)
        
        return csv_string
        
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        return f"Error: {str(e)}"

def export_data_to_json(data: Union[pd.DataFrame, List[Dict], Dict]) -> str:
    """Export data to JSON format"""
    try:
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict('records')
        else:
            json_data = data
        
        return json.dumps(json_data, indent=2, default=str)
        
    except Exception as e:
        print(f"Error exporting to JSON: {str(e)}")
        return f'{{"error": "{str(e)}"}}'

def filter_stocks_by_criteria(stocks: List[str], criteria: Dict[str, Any]) -> List[str]:
    """Filter stocks based on various criteria"""
    try:
        filtered_stocks = stocks.copy()
        
        # Filter by sector if specified
        if 'sector' in criteria and criteria['sector'] != 'All':
            # This would require a sector mapping - simplified for now
            pass
        
        # Filter by market cap if specified
        if 'market_cap' in criteria:
            # This would require market cap data - simplified for now
            pass
        
        # Filter by price range if specified
        if 'price_range' in criteria:
            # This would require current price data - simplified for now
            pass
        
        return filtered_stocks
        
    except Exception as e:
        print(f"Error filtering stocks: {str(e)}")
        return stocks

def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    try:
        current_time = datetime.now()
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        is_weekday = current_time.weekday() < 5
        
        market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open_time <= current_time <= market_close_time
        is_open = is_weekday and is_market_hours
        
        # Determine session
        if not is_weekday:
            session = 'weekend'
        elif current_time < market_open_time:
            session = 'pre_market'
        elif current_time > market_close_time:
            session = 'after_hours'
        else:
            session = 'market_hours'
        
        # Calculate time to next session
        if session == 'pre_market':
            next_session_time = market_open_time
            next_session = 'market_open'
        elif session == 'market_hours':
            next_session_time = market_close_time
            next_session = 'market_close'
        elif session == 'after_hours':
            next_day = current_time + timedelta(days=1)
            next_session_time = next_day.replace(hour=9, minute=15, second=0, microsecond=0)
            next_session = 'market_open'
        else:  # weekend
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 1
            next_monday = current_time + timedelta(days=days_until_monday)
            next_session_time = next_monday.replace(hour=9, minute=15, second=0, microsecond=0)
            next_session = 'market_open'
        
        time_to_next = next_session_time - current_time
        
        return {
            'is_open': is_open,
            'session': session,
            'current_time': current_time,
            'next_session': next_session,
            'next_session_time': next_session_time,
            'time_to_next': time_to_next,
            'status_text': '🟢 Market Open' if is_open else '🔴 Market Closed'
        }
        
    except Exception as e:
        print(f"Error getting market status: {str(e)}")
        return {
            'is_open': False,
            'session': 'unknown',
            'status_text': '⚠️ Status Unknown'
        }

def calculate_trading_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics"""
    try:
        if not trades:
            return get_empty_metrics()
        
        # Filter completed trades
        completed_trades = [t for t in trades if t.get('status') == 'closed' and 'pnl' in t]
        
        if not completed_trades:
            return get_empty_metrics()
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in completed_trades)
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = sum(abs(t['pnl']) for t in losing_trades) if losing_trades else 0
        
        # Ratios
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        
        # Risk metrics
        pnl_list = [t['pnl'] for t in completed_trades]
        max_drawdown = calculate_max_drawdown(pnl_list)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'best_trade': max(pnl_list) if pnl_list else 0,
            'worst_trade': min(pnl_list) if pnl_list else 0
        }
        
    except Exception as e:
        print(f"Error calculating trading metrics: {str(e)}")
        return get_empty_metrics()

def get_empty_metrics() -> Dict[str, Any]:
    """Return empty metrics"""
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0,
        'max_drawdown': 0,
        'best_trade': 0,
        'worst_trade': 0
    }

def calculate_max_drawdown(pnl_list: List[float]) -> float:
    """Calculate maximum drawdown"""
    try:
        if not pnl_list:
            return 0
        
        cumulative_pnl = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        
        return abs(min(drawdown)) if len(drawdown) > 0 else 0
        
    except Exception as e:
        print(f"Error calculating max drawdown: {str(e)}")
        return 0

def validate_trade_data(trade_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate trade data before processing"""
    try:
        required_fields = ['symbol', 'trade_type', 'entry_price', 'quantity']
        
        for field in required_fields:
            if field not in trade_data:
                return False, f"Missing required field: {field}"
        
        # Validate trade type
        if trade_data['trade_type'] not in ['BUY', 'SELL']:
            return False, "Trade type must be 'BUY' or 'SELL'"
        
        # Validate numeric fields
        numeric_fields = ['entry_price', 'quantity']
        for field in numeric_fields:
            if not isinstance(trade_data[field], (int, float)) or trade_data[field] <= 0:
                return False, f"{field} must be a positive number"
        
        # Validate symbol
        if not validate_stock_symbol(trade_data['symbol']):
            return False, "Invalid stock symbol"
        
        return True, "Valid trade data"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    try:
        diff = high - low
        levels = {}
        
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for ratio in fib_ratios:
            levels[f"{ratio:.3f}"] = high - (diff * ratio)
        
        return levels
        
    except Exception as e:
        print(f"Error calculating Fibonacci levels: {str(e)}")
        return {}

def calculate_support_resistance_levels(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """Calculate support and resistance levels"""
    try:
        if data.empty or len(data) < window:
            return {'support': [], 'resistance': []}
        
        # Simple pivot point method
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        # Find pivot highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))
        
        return {
            'resistance': resistance_levels[:5],  # Top 5
            'support': support_levels[-5:]       # Top 5
        }
        
    except Exception as e:
        print(f"Error calculating support/resistance: {str(e)}")
        return {'support': [], 'resistance': []}

def format_time_duration(seconds: int) -> str:
    """Format time duration in human readable format"""
    try:
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"
            
    except Exception as e:
        print(f"Error formatting time duration: {str(e)}")
        return f"{seconds}s"

def clean_stock_symbol(symbol: str) -> str:
    """Clean and standardize stock symbol"""
    try:
        if not symbol:
            return ""
        
        # Remove common suffixes and clean
        symbol = symbol.upper().strip()
        symbol = symbol.replace('.NS', '').replace('.BO', '')
        symbol = symbol.replace(' ', '')
        
        return symbol
        
    except Exception as e:
        print(f"Error cleaning stock symbol: {str(e)}")
        return symbol

def calculate_volatility(data: pd.DataFrame, period: int = 20) -> float:
    """Calculate price volatility"""
    try:
        if data.empty or len(data) < period:
            return 0.0
        
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=period).std().iloc[-1]
        
        # Annualized volatility
        return volatility * np.sqrt(252) * 100
        
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return 0.0

def is_trading_time() -> bool:
    """Check if current time is within trading hours"""
    try:
        current_time = datetime.now()
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        is_weekday = current_time.weekday() < 5
        
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open <= current_time <= market_close
        
        return is_weekday and is_market_hours
        
    except Exception:
        return False

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with fallback"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
    """Round price to valid tick size"""
    try:
        if tick_size <= 0:
            return price
        
        return round(price / tick_size) * tick_size
        
    except Exception:
        return price

