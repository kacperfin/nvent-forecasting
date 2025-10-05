# nVent Forecasting

An extensible ML model evaluation project with nested cross-validation. Built for order and sales forecasting at nVent.

## Features

- **Dataset management** - extend the DataHandler class with your custom transformations
- **Feature engineering for time series** - lag features and time-based features out of the box
- **Nested time series cross-validation** -  evaluate multiple models simultaneously, returns DataFrame for easy analysis
- **Aggregated metrics** - train on daily data, evaluate at daily, weekly, and monthly granularity
- **Visualization of cross-validation results** - one-line plotting of your model predictions at any time resolution
- **Extensible model architecture** - extend BaseForecaster to add your custom models

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```