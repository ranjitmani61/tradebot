"""
AI-powered signal generation module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AISignalGenerator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ML model"""
        try:
            if data.empty or len(data) < 20:
                return np.array([])
            
            # Import technical indicators here to avoid circular import
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            features = []
            
            # Price-based features
            if 'Close' in data.columns and len(data) >= 5:
                # Price momentum
                features.append(data['Close'].pct_change(5).iloc[-1] * 100)  # 5-period momentum
                features.append(data['Close'].pct_change(1).iloc[-1] * 100)  # 1-period momentum
                
                # Price position relative to moving averages
                sma_20 = tech_indicators.calculate_sma(data, 20)
                if len(sma_20.dropna()) > 0:
                    features.append((data['Close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100)
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0])
            
            # Volume-based features
            if 'Volume' in data.columns and len(data) >= 20:
                volume_ratio = tech_indicators.calculate_volume_ratio(data, 20)
                if len(volume_ratio.dropna()) > 0:
                    features.append(volume_ratio.iloc[-1])
                else:
                    features.append(1)
                
                # Volume momentum
                volume_change = data['Volume'].pct_change(1).iloc[-1]
                features.append(volume_change if not np.isnan(volume_change) else 0)
            else:
                features.extend([1, 0])
            
            # Technical indicators
            features.append(indicators.get('rsi', 50))
            features.append(indicators.get('macd', 0))
            features.append(indicators.get('macd_histogram', 0))
            
            # Bollinger Bands position
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            current_price = indicators.get('current_price', 0)
            
            if bb_upper > bb_lower and bb_upper > 0:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                features.append(bb_position)
            else:
                features.append(0.5)
            
            # Volatility (ATR normalized)
            atr = indicators.get('atr', 0)
            if current_price > 0 and atr > 0:
                features.append(atr / current_price * 100)
            else:
                features.append(1)
            
            # MACD signal strength
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            features.append(macd - macd_signal)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return np.array([]).reshape(1, -1)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate trading signal using rule-based approach"""
        try:
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            if data.empty or len(data) < 20:
                return self._get_default_signal(symbol)
            
            # Calculate technical indicators
            indicators = tech_indicators.calculate_all_indicators(data)
            
            if not indicators:
                return self._get_default_signal(symbol)
            
            # Rule-based signal generation
            signal_score = 0
            confidence_factors = []
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                signal_score += 2
                confidence_factors.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:  # Overbought
                signal_score -= 2
                confidence_factors.append(f"RSI overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:  # Neutral
                signal_score += 0.5
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > 0:  # Bullish
                signal_score += 1.5
                confidence_factors.append("MACD bullish crossover")
            elif macd < macd_signal and macd_histogram < 0:  # Bearish
                signal_score -= 1.5
                confidence_factors.append("MACD bearish crossover")
            
            # Moving Average signals
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            
            if current_price > sma_20 and sma_20 > 0:  # Above SMA
                signal_score += 1
                confidence_factors.append("Price above SMA 20")
            elif current_price < sma_20 and sma_20 > 0:  # Below SMA
                signal_score -= 1
                confidence_factors.append("Price below SMA 20")
            
            # Bollinger Bands signals
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            if current_price <= bb_lower and bb_lower > 0:  # At lower band
                signal_score += 1.5
                confidence_factors.append("Price at BB lower band")
            elif current_price >= bb_upper and bb_upper > 0:  # At upper band
                signal_score -= 1.5
                confidence_factors.append("Price at BB upper band")
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:  # High volume
                signal_score *= 1.2
                confidence_factors.append(f"High volume ({volume_ratio:.1f}x)")
            elif volume_ratio < 0.5:  # Low volume
                signal_score *= 0.8
            
            # Price momentum
            price_change = indicators.get('price_change', 0)
            if abs(price_change) > 2:  # Strong momentum
                if price_change > 0:
                    signal_score += 0.5
                else:
                    signal_score -= 0.5
                confidence_factors.append(f"Strong momentum ({price_change:+.1f}%)")
            
            # Determine final signal
            if signal_score >= 2:
                signal = 'BUY'
                confidence = min(85, 50 + abs(signal_score) * 8)
            elif signal_score <= -2:
                signal = 'SELL'
                confidence = min(85, 50 + abs(signal_score) * 8)
            else:
                signal = 'HOLD'
                confidence = 50 + abs(signal_score) * 5
            
            # Ensure confidence is reasonable
            confidence = max(30, min(95, confidence))
            
            return {
                'signal': signal,
                'confidence': round(confidence, 1),
                'price': current_price,
                'rsi': rsi,
                'macd': macd,
                'volume': indicators.get('current_volume', 0),
                'volume_ratio': volume_ratio,
                'signal_score': signal_score,
                'factors': confidence_factors[:3],  # Top 3 factors
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol
            }
            
        except Exception as e:
            print(f"Error generating signal for {symbol}: {str(e)}")
            return self._get_default_signal(symbol)
    
    def _get_default_signal(self, symbol: str) -> Dict[str, Any]:
        """Return default signal when calculation fails"""
        return {
            'signal': 'HOLD',
            'confidence': 50.0,
            'price': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'volume': 0,
            'volume_ratio': 1.0,
            'signal_score': 0,
            'factors': ['Insufficient data'],
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol
        }
    
    def batch_generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Generate signals for multiple stocks"""
        try:
            signals = {}
            
            for symbol, data in data_dict.items():
                signals[symbol] = self.generate_signal(data, symbol)
            
            return signals
            
        except Exception as e:
            print(f"Error in batch signal generation: {str(e)}")
            return {}
    
    def get_signal_strength(self, signal_data: Dict[str, Any]) -> str:
        """Get signal strength description"""
        try:
            confidence = signal_data.get('confidence', 50)
            
            if confidence >= 80:
                return "Very Strong"
            elif confidence >= 70:
                return "Strong"
            elif confidence >= 60:
                return "Moderate"
            elif confidence >= 50:
                return "Weak"
            else:
                return "Very Weak"
                
        except Exception:
            return "Unknown"
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validate if signal is reliable"""
        try:
            confidence = signal_data.get('confidence', 0)
            signal = signal_data.get('signal', 'HOLD')
            
            # Minimum confidence threshold
            if confidence < 60 and signal != 'HOLD':
                return False
            
            # Check if key indicators are available
            required_fields = ['price', 'rsi', 'macd']
            for field in required_fields:
                if field not in signal_data or signal_data[field] == 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_risk_assessment(self, data: pd.DataFrame, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for the generated signal"""
        try:
            if data.empty:
                return {'risk_level': 'High', 'risk_score': 80}
            
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # ATR-based risk
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            atr = tech_indicators.calculate_atr(data)
            
            if len(atr.dropna()) > 0:
                atr_pct = (atr.iloc[-1] / data['Close'].iloc[-1]) * 100
            else:
                atr_pct = 2  # Default
            
            # Risk score calculation
            risk_score = 0
            
            # Volatility component
            if volatility > 40:
                risk_score += 30
            elif volatility > 25:
                risk_score += 20
            else:
                risk_score += 10
            
            # ATR component
            if atr_pct > 5:
                risk_score += 25
            elif atr_pct > 3:
                risk_score += 15
            else:
                risk_score += 5
            
            # Signal confidence component
            confidence = signal_data.get('confidence', 50)
            risk_score += (100 - confidence) * 0.3
            
            # Volume component
            volume_ratio = signal_data.get('volume_ratio', 1)
            if volume_ratio < 0.5:  # Low volume
                risk_score += 20
            elif volume_ratio > 2:  # Very high volume
                risk_score += 10
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = 'High'
            elif risk_score >= 50:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'risk_level': risk_level,
                'risk_score': min(100, max(0, risk_score)),
                'volatility': volatility,
                'atr_percentage': atr_pct
            }
            
        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")
            return {'risk_level': 'High', 'risk_score': 80}
