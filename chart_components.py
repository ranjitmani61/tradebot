"""
Chart components for visualization
"""

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

class ChartComponents:
    def __init__(self):
        self.default_height = 600
        
    def create_candlestick_chart(self, data: pd.DataFrame, symbol: str, 
                               indicators: Dict[str, Any] = None) -> go.Figure:
        """Create candlestick chart with indicators"""
        try:
            if data.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create subplots with proper row heights
            fig = sp.make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{symbol} - Price Action', 'Volume', 'RSI'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            try:
                sma_20 = tech_indicators.calculate_sma(data, 20)
                if len(sma_20.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=sma_20,
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ),
                        row=1, col=1
                    )
                
                sma_50 = tech_indicators.calculate_sma(data, 50)
                if len(sma_50.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=sma_50,
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                
                # Bollinger Bands
                bb_data = tech_indicators.calculate_bollinger_bands(data)
                if len(bb_data['bb_upper'].dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=bb_data['bb_upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=bb_data['bb_lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            except Exception as e:
                print(f"Error adding indicators to chart: {str(e)}")
            
            # Volume chart
            colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # RSI chart
            try:
                rsi = tech_indicators.calculate_rsi(data)
                if len(rsi.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=rsi,
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=3, col=1
                    )
                    
                    # RSI reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            except Exception as e:
                print(f"Error adding RSI to chart: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Technical Analysis',
                xaxis_rangeslider_visible=False,
                height=self.default_height,
                showlegend=True,
                legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating candlestick chart: {str(e)}")
            # Return empty figure on error
            fig = go.Figure()
            fig.add_annotation(text=f"Chart Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_volume_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volume chart"""
        try:
            if data.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Color bars based on price movement
            colors = []
            for i in range(len(data)):
                if data['Close'].iloc[i] >= data['Open'].iloc[i]:
                    colors.append('green')
                else:
                    colors.append('red')
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors
                )
            )
            
            # Add volume moving average
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            try:
                volume_sma = tech_indicators.calculate_volume_sma(data, 20)
                if len(volume_sma.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=volume_sma,
                            mode='lines',
                            name='Volume SMA 20',
                            line=dict(color='orange', width=2)
                        )
                    )
            except Exception:
                pass
            
            fig.update_layout(
                title=f'{symbol} - Volume Analysis',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volume chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Chart Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_indicator_chart(self, data: pd.DataFrame, indicator_type: str, symbol: str) -> go.Figure:
        """Create individual indicator chart"""
        try:
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            
            fig = go.Figure()
            
            if indicator_type == 'RSI':
                rsi = tech_indicators.calculate_rsi(data)
                if len(rsi.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=rsi,
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        )
                    )
                    
                    # Reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.add_hline(y=50, line_dash="dot", line_color="gray")
                    
                    fig.update_yaxes(range=[0, 100])
            
            elif indicator_type == 'MACD':
                macd_data = tech_indicators.calculate_macd(data)
                if len(macd_data['macd'].dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=macd_data['macd'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=macd_data['macd_signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=2)
                        )
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=macd_data['macd_histogram'],
                            name='Histogram',
                            marker_color='gray',
                            opacity=0.7
                        )
                    )
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            elif indicator_type == 'Stochastic':
                stoch_data = tech_indicators.calculate_stochastic(data)
                if len(stoch_data['stoch_k'].dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=stoch_data['stoch_k'],
                            mode='lines',
                            name='%K',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=stoch_data['stoch_d'],
                            mode='lines',
                            name='%D',
                            line=dict(color='red', width=2)
                        )
                    )
                    
                    # Reference lines
                    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.add_hline(y=50, line_dash="dot", line_color="gray")
                    
                    fig.update_yaxes(range=[0, 100])
            
            elif indicator_type == 'Williams %R':
                williams_r = tech_indicators.calculate_williams_r(data)
                if len(williams_r.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=williams_r,
                            mode='lines',
                            name='Williams %R',
                            line=dict(color='orange', width=2)
                        )
                    )
                    
                    # Reference lines
                    fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.add_hline(y=-50, line_dash="dot", line_color="gray")
                    
                    fig.update_yaxes(range=[-100, 0])
            
            elif indicator_type == 'CCI':
                cci = tech_indicators.calculate_cci(data)
                if len(cci.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=cci,
                            mode='lines',
                            name='CCI',
                            line=dict(color='cyan', width=2)
                        )
                    )
                    
                    # Reference lines
                    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            elif indicator_type == 'ATR':
                atr = tech_indicators.calculate_atr(data)
                if len(atr.dropna()) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=atr,
                            mode='lines',
                            name='ATR',
                            line=dict(color='brown', width=2)
                        )
                    )
            
            else:
                fig.add_annotation(text=f"Indicator '{indicator_type}' not implemented", 
                                 x=0.5, y=0.5, showarrow=False)
            
            fig.update_layout(
                title=f'{symbol} - {indicator_type}',
                xaxis_title='Date',
                yaxis_title=indicator_type,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating {indicator_type} chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Chart Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_comparison_chart(self, data_dict: Dict[str, pd.DataFrame], symbols: List[str]) -> go.Figure:
        """Create comparison chart for multiple symbols"""
        try:
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, symbol in enumerate(symbols):
                if symbol in data_dict and not data_dict[symbol].empty:
                    data = data_dict[symbol]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                            y=data['Close'],
                            mode='lines',
                            name=symbol,
                            line=dict(color=color, width=2)
                        )
                    )
            
            fig.update_layout(
                title='Price Comparison',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating comparison chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Chart Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str, chart_type: str = 'line') -> go.Figure:
        """Create basic price chart"""
        try:
            if data.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
                return fig
            
            fig = go.Figure()
            
            if chart_type == 'candlestick':
                fig.add_trace(
                    go.Candlestick(
                        x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    )
                )
            else:  # line chart
                fig.add_trace(
                    go.Scatter(
                        x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                        y=data['Close'],
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    )
                )
            
            fig.update_layout(
                title=f'{symbol} - Price Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Chart Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
