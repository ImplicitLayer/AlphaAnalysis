__version__ = "0.1.0"
__author__ = "ArtemBurenok"
__email__ = "burenok023@gmail.com"

# Open Source
from .data import data_loader, data_cleaning
from .visualizations import plots, indicators
from .models import classical_models, ml_models
from .portfolio import risk_analysis, clustering
from .signal_generation import technical_signals

__all__ = [
    "data_loader", "data_cleaning", "plots", "indicators",
    "classical_models", "ml_models", "risk_analysis", "clustering", "technical_signals"
]

__all__.extend([
    "feature_engineering", "deep_learning", "auto_ml",
    "backtesting", "risk_management", "optimization",
    "fundamental_signals", "sentiment_analysis"
])

