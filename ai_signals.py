"""
Real-time intraday AI trading signals with specific buy/sell price recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class IntradayAISignals:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.price_targets = {}
        self.active_alerts = {}
        
    def analyze_realtime_movement(self, data: pd.DataFrame, symbol: str, 
                                 current_price: float, seconds_lookback: int = 30) -> Dict[str, Any]:
        """
        Analyze second-by-second price movements and generate buy/sell price alerts
        """
        try:
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            if data.empty or len(data) < seconds_lookback:
                return self._get_default_alert(symbol, current_price)
            
            # Get recent price movements (last X seconds)
            recent_data = data.tail(seconds_lookback)
            
            # Calculate price momentum
            price_change = self._calculate_price_momentum(recent_data)
            volatility = self._calculate_short_term_volatility(recent_data)
            
            # Calculate technical indicators for context
            indicators = self._calculate_quick_indicators(tech_indicators, data)
            
            # Generate specific buy/sell price recommendations
            signal_data = self._generate_intraday_signal(
                symbol, current_price, price_change, volatility, indicators, recent_data
            )
            
            return signal_data
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            return self._get_default_alert(symbol, current_price)
    
    def _calculate_price_momentum(self, recent_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price momentum over recent seconds"""
        try:
            if len(recent_data) < 2:
                return {'direction': 0, 'strength': 0, 'acceleration': 0}
            
            prices = recent_data['Close'].values
            
            # Direction: positive if trending up, negative if down
            direction = (prices[-1] - prices[0]) / prices[0] * 100
            
            # Strength: how consistent the movement is
            price_changes = np.diff(prices)
            consistent_moves = np.sum(np.sign(price_changes) == np.sign(direction))
            strength = consistent_moves / len(price_changes) if len(price_changes) > 0 else 0
            
            # Acceleration: is the movement getting stronger?
            if len(prices) >= 3:
                recent_change = (prices[-1] - prices[-2]) / prices[-2] * 100
                earlier_change = (prices[-2] - prices[-3]) / prices[-3] * 100
                acceleration = recent_change - earlier_change
            else:
                acceleration = 0
            
            return {
                'direction': direction,
                'strength': strength,
                'acceleration': acceleration
            }
            
        except Exception:
            return {'direction': 0, 'strength': 0, 'acceleration': 0}
    
    def _calculate_short_term_volatility(self, recent_data: pd.DataFrame) -> float:
        """Calculate volatility over recent seconds"""
        try:
            if len(recent_data) < 2:
                return 0
            
            price_changes = recent_data['Close'].pct_change().dropna()
            return price_changes.std() * 100 if len(price_changes) > 0 else 0
            
        except Exception:
            return 0
    
    def _calculate_quick_indicators(self, tech_indicators, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate essential technical indicators quickly"""
        try:
            if len(data) < 14:
                return {}
            
            # RSI for overbought/oversold
            rsi_series = tech_indicators.calculate_rsi(data, period=14)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
            
            # Short-term moving averages
            sma_5 = tech_indicators.calculate_sma(data, 5)
            sma_5_val = float(sma_5.iloc[-1]) if not sma_5.empty else 0
            
            sma_10 = tech_indicators.calculate_sma(data, 10)
            sma_10_val = float(sma_10.iloc[-1]) if not sma_10.empty else 0
            
            # Current price position
            current_price = float(data['Close'].iloc[-1])
            
            # Support and resistance from recent data
            recent_high = float(data['High'].tail(20).max())
            recent_low = float(data['Low'].tail(20).min())
            
            return {
                'rsi': rsi,
                'sma_5': sma_5_val,
                'sma_10': sma_10_val,
                'current_price': current_price,
                'recent_high': recent_high,
                'recent_low': recent_low
            }
            
        except Exception:
            return {}
    
    def _generate_intraday_signal(self, symbol: str, current_price: float, 
                                 momentum: Dict[str, float], volatility: float,
                                 indicators: Dict[str, Any], recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate specific buy/sell price recommendations for intraday trading"""
        
        direction = momentum.get('direction', 0)
        strength = momentum.get('strength', 0)
        acceleration = momentum.get('acceleration', 0)
        
        rsi = indicators.get('rsi', 50)
        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        recent_high = indicators.get('recent_high', current_price)
        recent_low = indicators.get('recent_low', current_price)
        
        # Calculate price targets based on volatility and momentum
        volatility_buffer = max(volatility * 0.01, 0.002)  # Minimum 0.2% buffer
        
        alert_type = "HOLD"
        buy_price = None
        sell_price = None
        confidence = 50
        reasoning = []
        
        # UPWARD MOVEMENT DETECTED
        if direction > 0.1 and strength > 0.6:  # Strong upward movement
            if acceleration > 0:  # Accelerating upward
                alert_type = "BUY_MOMENTUM"
                # Buy at current price or slight pullback
                buy_price = round(current_price * (1 - volatility_buffer), 4)
                # Sell target: current resistance or momentum target
                sell_price = round(current_price * (1 + volatility_buffer * 2), 4)
                confidence = min(85, 60 + strength * 25)
                reasoning.append(f"Strong upward momentum ({direction:+.2f}%)")
                reasoning.append(f"Accelerating movement ({acceleration:+.3f}%)")
                
            elif rsi < 70:  # Not overbought yet
                alert_type = "BUY_TREND"
                buy_price = round(current_price * (1 - volatility_buffer * 0.5), 4)
                sell_price = round(min(recent_high * 0.99, current_price * (1 + volatility_buffer * 1.5)), 4)
                confidence = min(75, 55 + strength * 20)
                reasoning.append(f"Upward trend ({direction:+.2f}%)")
                reasoning.append(f"RSI not overbought ({rsi:.1f})")
        
        # DOWNWARD MOVEMENT DETECTED  
        elif direction < -0.1 and strength > 0.6:  # Strong downward movement
            if acceleration < 0:  # Accelerating downward
                alert_type = "SELL_MOMENTUM"
                # Sell at current price or slight bounce
                sell_price = round(current_price * (1 + volatility_buffer), 4)
                # Buy back target: support level or oversold bounce
                buy_price = round(current_price * (1 - volatility_buffer * 2), 4)
                confidence = min(85, 60 + strength * 25)
                reasoning.append(f"Strong downward momentum ({direction:+.2f}%)")
                reasoning.append(f"Accelerating downward ({acceleration:+.3f}%)")
                
            elif rsi > 30:  # Not oversold yet
                alert_type = "SELL_TREND"
                sell_price = round(current_price * (1 + volatility_buffer * 0.5), 4)
                buy_price = round(max(recent_low * 1.01, current_price * (1 - volatility_buffer * 1.5)), 4)
                confidence = min(75, 55 + strength * 20)
                reasoning.append(f"Downward trend ({direction:+.2f}%)")
                reasoning.append(f"RSI not oversold ({rsi:.1f})")
        
        # SIDEWAYS/RANGE TRADING
        elif abs(direction) <= 0.1 and volatility > 0.5:
            if current_price <= recent_low * 1.02:  # Near support
                alert_type = "BUY_SUPPORT"
                buy_price = round(current_price, 4)
                sell_price = round(recent_high * 0.98, 4)
                confidence = 65
                reasoning.append("Price near support level")
                reasoning.append("Range trading opportunity")
                
            elif current_price >= recent_high * 0.98:  # Near resistance
                alert_type = "SELL_RESISTANCE"
                sell_price = round(current_price, 4)
                buy_price = round(recent_low * 1.02, 4)
                confidence = 65
                reasoning.append("Price near resistance level")
                reasoning.append("Range trading opportunity")
        
        # BREAKOUT DETECTION
        if current_price > recent_high * 1.001:  # Breaking resistance
            alert_type = "BUY_BREAKOUT"
            buy_price = round(current_price, 4)
            sell_price = round(current_price * (1 + volatility_buffer * 3), 4)
            confidence = 80
            reasoning.append("Breaking above resistance")
            reasoning.append("Potential breakout")
            
        elif current_price < recent_low * 0.999:  # Breaking support
            alert_type = "SELL_BREAKDOWN"
            sell_price = round(current_price, 4)
            buy_price = round(current_price * (1 - volatility_buffer * 3), 4)
            confidence = 80
            reasoning.append("Breaking below support")
            reasoning.append("Potential breakdown")
        
        # Calculate stop loss levels
        stop_loss_buy = round(current_price * (1 - volatility_buffer * 2), 4) if buy_price else None
        stop_loss_sell = round(current_price * (1 + volatility_buffer * 2), 4) if sell_price else None
        
        return {
            'symbol': symbol,
            'alert_type': alert_type,
            'current_price': round(current_price, 4),
            'buy_price': buy_price,
            'sell_price': sell_price,
            'stop_loss_buy': stop_loss_buy,
            'stop_loss_sell': stop_loss_sell,
            'confidence': round(confidence, 1),
            'reasoning': reasoning[:3],  # Top 3 reasons
            'momentum': {
                'direction': round(direction, 3),
                'strength': round(strength, 3),
                'acceleration': round(acceleration, 4)
            },
            'volatility': round(volatility, 3),
            'rsi': round(rsi, 1),
            'timestamp': pd.Timestamp.now(),
            'timeframe': 'intraday'
        }
    
    def get_price_alerts(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame], 
                        seconds_lookback: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get real-time price alerts for multiple symbols"""
        alerts = {}
        
        for symbol in symbols:
            if symbol in data_dict and not data_dict[symbol].empty:
                current_price = float(data_dict[symbol]['Close'].iloc[-1])
                alert = self.analyze_realtime_movement(
                    data_dict[symbol], symbol, current_price, seconds_lookback
                )
                alerts[symbol] = alert
        
        return alerts
    
    def format_trading_alert(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data into readable trading message"""
        symbol = alert_data['symbol']
        alert_type = alert_data['alert_type']
        current_price = alert_data['current_price']
        buy_price = alert_data.get('buy_price')
        sell_price = alert_data.get('sell_price')
        confidence = alert_data['confidence']
        reasoning = alert_data.get('reasoning', [])
        
        if alert_type == "HOLD":
            return f"📊 {symbol} @ ${current_price} - HOLD - No clear signal"
        
        message = f"🚨 {symbol} ALERT @ ${current_price}\n"
        message += f"📈 Signal: {alert_type.replace('_', ' ')}\n"
        message += f"🎯 Confidence: {confidence}%\n"
        
        if buy_price:
            message += f"💰 BUY at: ${buy_price}\n"
        if sell_price:
            message += f"💸 SELL at: ${sell_price}\n"
            
        if alert_data.get('stop_loss_buy'):
            message += f"🛑 Stop Loss (Buy): ${alert_data['stop_loss_buy']}\n"
        if alert_data.get('stop_loss_sell'):
            message += f"🛑 Stop Loss (Sell): ${alert_data['stop_loss_sell']}\n"
        
        if reasoning:
            message += f"📋 Reasons: {', '.join(reasoning)}\n"
        
        return message
    
    def _get_default_alert(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Return default alert when analysis fails"""
        return {
            'symbol': symbol,
            'alert_type': 'HOLD',
            'current_price': round(current_price, 4),
            'buy_price': None,
            'sell_price': None,
            'stop_loss_buy': None,
            'stop_loss_sell': None,
            'confidence': 50.0,
            'reasoning': ['Insufficient data'],
            'momentum': {'direction': 0, 'strength': 0, 'acceleration': 0},
            'volatility': 0,
            'rsi': 50,
            'timestamp': pd.Timestamp.now(),
            'timeframe': 'intraday'
        }
