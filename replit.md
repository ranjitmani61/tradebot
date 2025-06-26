# Professional Intraday Trading Assistant

## Overview

This is a comprehensive professional intraday trading assistant built with Python and Streamlit. The application provides real-time NSE/BSE stock market analysis, AI-powered signal generation, automated scanning, and multi-channel alert systems. It's designed as a complete trading dashboard for intraday traders focusing on Indian stock markets.

## System Architecture

The application follows a modular microservices architecture with clear separation of concerns:

- **Frontend Layer**: Streamlit-based web interface with responsive design and real-time updates
- **Data Layer**: Real-time market data fetching using yfinance API with intelligent caching
- **Analysis Engine**: Technical indicator calculations using custom algorithms and fallback implementations
- **AI/ML Layer**: Rule-based signal generation with enhanced confidence scoring
- **Alert System**: Multi-channel notification system via Telegram integration
- **Persistence Layer**: Trading journal and performance tracking with session state management
- **Configuration Management**: Centralized configuration for stocks, sectors, and trading parameters

The system is designed as a real-time trading dashboard that can run continuously with auto-refresh capabilities and supports both NSE and BSE stock exchanges.

## Key Components

### Core Modules

1. **Main Application** (`app.py`)
   - Streamlit-based orchestrator managing all system components
   - Session state management and error handling
   - Modular import system with safe fallbacks

2. **Data Fetcher** (`data_fetcher.py`)
   - Real-time market data retrieval using yfinance API
   - Intelligent caching mechanism (5-minute TTL)
   - Support for both NSE (.NS) and BSE (.BO) exchanges
   - Retry mechanism with exponential backoff

3. **Technical Indicators** (`technical_indicators.py`)
   - Comprehensive technical analysis with fallback implementations
   - RSI, MACD, moving averages, and volume indicators
   - Custom calculations when external libraries are unavailable

4. **AI Signal Generator** (`ai_signals.py`)
   - Enhanced rule-based signal generation system
   - Multi-factor analysis combining RSI, MACD, volume, and trend indicators
   - Confidence scoring and signal validation

5. **Stock Scanner** (`scanner.py`)
   - Real-time market scanning for trading opportunities
   - Volume surge detection and price breakout identification
   - Multi-stock analysis with performance ranking

6. **Chart Components** (`chart_components.py`)
   - Interactive candlestick charts with technical overlays
   - Multi-panel layout with volume and indicator subplots
   - Plotly-based visualization with professional styling

7. **Telegram Integration** (`telegram_bot.py`)
   - Real-time alert delivery via Telegram
   - Command processing and status updates
   - Session state management for connection status

8. **Trading Journal** (`trading_journal.py`)
   - Trade tracking and performance analysis
   - P&L calculations and trade statistics
   - Export capabilities for external analysis

9. **UI Components** (`ui_components.py`)
   - Reusable UI elements and layout management
   - Session state initialization and management
   - Metric cards and dashboard components

## Data Flow

1. **Market Data Acquisition**: yfinance API fetches real-time data for selected stocks
2. **Technical Analysis**: Indicators are calculated using custom algorithms with fallback methods
3. **Signal Generation**: AI module processes indicators to generate buy/sell/hold signals
4. **Market Scanning**: Scanner module analyzes multiple stocks for opportunities
5. **Visualization**: Charts display price action with technical overlays
6. **Alert Distribution**: Signals are sent via Telegram for real-time notifications
7. **Performance Tracking**: Trading journal records and analyzes trade performance

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **yfinance**: Real-time market data acquisition from Yahoo Finance
- **Plotly**: Interactive charting and visualization
- **Pandas/NumPy**: Data manipulation and numerical computations
- **Requests**: HTTP client for API communications

### Optional Enhancements
- **TA-Lib**: Advanced technical analysis (with fallback implementations)
- **Scikit-learn**: Machine learning capabilities for signal enhancement

### Configuration
- **NSE/BSE Stock Universe**: Comprehensive list of 200+ Indian stocks
- **Sector Categorization**: Banking, IT, Auto, Pharma, FMCG, Energy sectors
- **Technical Parameters**: Configurable periods and thresholds for indicators

## Deployment Strategy

The application is configured for deployment on Replit with the following setup:

- **Runtime**: Python 3.11 environment
- **Server**: Streamlit running on port 5000
- **Autoscaling**: Configured for automatic scaling based on demand
- **Dependencies**: Managed via pyproject.toml with locked versions in uv.lock

### Environment Variables
- `TELEGRAM_BOT_TOKEN`: For alert notifications
- `TELEGRAM_CHAT_ID`: Target chat for alerts

### Performance Optimization
- Streamlit caching with 5-minute TTL for market data
- Session state management for user preferences
- Concurrent data fetching for multiple stocks
- Fallback implementations for robust operation

## Changelog
- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.