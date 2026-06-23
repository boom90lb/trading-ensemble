"""Statistical-arbitrage research primitives.

This package is deliberately separate from the single-symbol forecast ensemble:
arbitrage needs cross-asset selection, hedge-aware positions, and portfolio-level
accounting rather than independent ticker forecasts.

Two research paths live here: pairs (Engle-Granger cointegration, frozen per-fold
candidates) and residual (Avellaneda-Lee PCA eigenportfolio residuals, causally
rolling estimators).
"""

__all__: list[str] = []
