# src/models/ensemble.py
"""Ensemble model. Emits unified position outputs in [-1, 1].

Forecast members (ARIMA, Prophet, LSTM, XGBoost) produce ŷ_{t+h}; policy
members (LSTM-PPO, xLSTM-PPO, xLSTM-GRPO) produce positions directly.
Forecast outputs are mapped through inverse-volatility sizing before
weighting so the ensemble averages dimensionally coherent quantities (B8).
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import (  # type: ignore
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import MODELS_DIR, EnsembleConfig
from src.conformal import ACIState, EnbPICalibrator
from src.models.base import BaseModel
from src.models.lstm_ppo import LSTMPPO
from src.models.mapping import forecast_to_position, ideal_position, realized_vol
from src.models.registry import ModelKind, is_forecast, is_policy, model_kind
from src.validation.walk_forward import PurgedWalkForward

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple forecasting models."""

    def __init__(self, target_column: str = "close", horizon: int = 5, config: Optional[EnsembleConfig] = None):
        """Initialize the ensemble model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            config: Ensemble configuration
        """
        super().__init__(name="ensemble", target_column=target_column, horizon=horizon)

        # Store configuration
        self.config = config or EnsembleConfig(models=[])

        self.models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}
        self.errors: Dict[str, float] = {}
        self.kinds: Dict[str, ModelKind] = {}
        self.is_fitted = False

        # Vol-targeting and meta-learner knobs. Held as attrs so they can be
        # overridden after construction without re-instantiating the ensemble.
        self.vol_window: int = 20
        self.vol_floor: float = 5e-3
        # Read vol-targeting knobs from config (Phase 2.7) so a sweep can vary
        # them; default to 1.0 when the config predates these fields.
        self.target_vol: float = getattr(self.config, "target_vol", 1.0)
        self.position_cap: float = getattr(self.config, "position_cap", 1.0)
        # Phase 2.1: PurgedWalkForward params for meta-learner OOF.
        # Used for both forecast members (was sklearn KFold) and policy
        # members (was in-sample-leaky train-tail holdout via
        # meta_policy_holdout, now retired).
        self.meta_cv_folds: int = 3
        self.meta_embargo_pct: float = 0.01

        # Phase 2.5: position-level conformal layer over OOF residuals.
        # Populated by fit() when a meta-learner is configured (dynamic
        # weighting). predict_band() falls back to ±position_cap bands when
        # conformal is None (e.g. static-weighted ensembles).
        self.conformal_alpha_target: float = 0.10
        self.conformal_aci_gamma: float = 0.01
        self.conformal: Optional[EnbPICalibrator] = None
        self.aci: Optional[ACIState] = None

        self.lstm_ppo_save_path: Optional[Path] = None
        self.xlstm_ppo_save_path: Optional[Path] = None
        self.xlstm_grpo_save_path: Optional[Path] = None

        for model_config in self.config.models:
            if model_config.enabled:
                model_instance = self._create_model(model_config.name)
                if model_instance:
                    self.models[model_config.name] = model_instance
                    self.weights[model_config.name] = model_config.weight
                    self.kinds[model_config.name] = model_kind(model_config.name)
                else:
                    logger.warning(f"Failed to create model '{model_config.name}', excluding from ensemble.")

        self._normalize_weights()

    @property
    def required_history(self) -> int:
        """Max trailing rows any member needs to score the latest bar.

        The ensemble must be fed enough history for its hungriest member
        (e.g. an RL agent's window_size+2), so the trading loop slices the
        trailing ``required_history`` rows. Point-only ensembles → 1.
        """
        if not self.models:
            return 1
        return max(m.required_history for m in self.models.values())

    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1."""
        if not self.weights:
            return

        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight

    def _create_model(self, model_name: str, seed: Optional[int] = None) -> Optional[BaseModel]:
        """Create a model instance by name.

        Args:
            model_name: Name of the model to create
            seed: Optional PRNG seed for policy (RL) members. When ``None``
                (the default, used by the production fit + OOF refit paths) the
                agent's own constructor default applies — no behavior change.
                The multi-seed eval driver (Phase 3.2) passes an explicit seed
                so each RL agent's ``jax.random.key(seed)`` differs per trial.
                Ignored by forecast members, which are deterministic given data.

        Returns:
            Model instance or None if not supported
        """
        try:
            if model_name == "arima":
                from src.models.arima import ARIMAModel

                return ARIMAModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "prophet":
                from src.models.prophet import ProphetModel

                return ProphetModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "lstm":
                from src.models.lstm import LSTMModel

                return LSTMModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "xgboost":
                from src.models.xgboost_model import XGBoostModel

                return XGBoostModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "lstm_ppo":
                # LSTM-PPO is now created directly
                policy_kwargs = {"seed": seed} if seed is not None else {}
                return LSTMPPO(target_column=self.target_column, horizon=self.horizon, **policy_kwargs)
            elif model_name == "xlstm_ppo":
                # Import the XLSTMPPOAgent model
                from src.models.xlstm_ppo import XLSTMPPOAgent

                policy_kwargs = {"seed": seed} if seed is not None else {}
                return XLSTMPPOAgent(target_column=self.target_column, horizon=self.horizon, **policy_kwargs)
            elif model_name == "xlstm_grpo":
                # Import the XLSTMGRPOAgent model
                from src.models.xlstm_grpo import XLSTMGRPOAgent

                policy_kwargs = {"seed": seed} if seed is not None else {}
                return XLSTMGRPOAgent(target_column=self.target_column, horizon=self.horizon, **policy_kwargs)
            else:
                logger.warning(f"Unsupported model: {model_name}")
                return None
        except ImportError as e:
            logger.error(f"Could not import {model_name} model: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating {model_name} model: {e}")
            return None

    def _fit_policy_member(
        self,
        name: str,
        model: BaseModel,
        X_with_close: pd.DataFrame,
        y: pd.Series,
        kwargs: Dict[str, Any],
        persist: bool,
    ) -> bool:
        """Fit one policy member. Centralizes the per-policy dispatch so
        the OOF refit (Phase 2.1) and the top-level fit share one code
        path. `persist=True` writes the model's save artifact and stashes
        the path on self; OOF folds set persist=False.
        """
        features = kwargs.get(f"{name}_features")
        if features is None:
            logger.error(f"Missing '{name}_features' kwarg; skipping {name}.")
            return False
        model.features = features  # type: ignore[attr-defined]

        if name == "lstm_ppo":
            timesteps = kwargs.get("lstm_ppo_timesteps", 100000)
            if not isinstance(model, LSTMPPO):
                logger.error(f"{name} model is not an LSTMPPO instance.")
                return False
            model.fit(X_with_close, y, total_timesteps=timesteps)
            if persist:
                self.lstm_ppo_save_path = model.save()
        elif name == "xlstm_ppo":
            from src.models.xlstm_ppo import XLSTMPPOAgent

            timesteps = kwargs.get("xlstm_ppo_timesteps", 100000)
            if not isinstance(model, XLSTMPPOAgent):
                logger.error(f"{name} model is not an XLSTMPPOAgent instance.")
                return False
            model.fit(X_with_close, y, total_timesteps=timesteps)
            if persist:
                self.xlstm_ppo_save_path = model.save()
        elif name == "xlstm_grpo":
            from src.models.xlstm_grpo import XLSTMGRPOAgent

            updates = kwargs.get("xlstm_grpo_updates", 1000)
            if not isinstance(model, XLSTMGRPOAgent):
                logger.error(f"{name} model is not an XLSTMGRPOAgent instance.")
                return False
            model.fit(X_with_close, y, total_updates=updates)
            if persist:
                self.xlstm_grpo_save_path = model.save()
        else:
            logger.error(f"Unknown policy member '{name}' in dispatch.")
            return False
        return True

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "EnsembleModel":
        """Fit the ensemble model by fitting each component model.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for specific model fit methods
                      (e.g., lstm_ppo_timesteps, lstm_ppo_features, lstm_ppo_X_train_with_close)

        Returns:
            Fitted ensemble model (self)
        """
        fitted_models: List[str] = []

        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name} model")
                if is_policy(name):
                    x_full = kwargs.get(f"{name}_X_train_with_close")
                    if x_full is None:
                        logger.error(
                            f"Missing '{name}_X_train_with_close' kwarg; skipping {name}."
                        )
                        continue
                    if not self._fit_policy_member(name, model, x_full, y_train, kwargs, persist=True):
                        continue
                else:
                    model.fit(X_train, y_train)

                fitted_models.append(name)
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}", exc_info=True)

        if self.config.weighting_strategy == "dynamic":
            self._fit_meta_learner_weights(X_train, y_train, fitted_models, kwargs)

        self._normalize_weights()

        self.is_fitted = len(fitted_models) > 0
        logger.info(f"Ensemble model fitted with {len(fitted_models)} component models")
        logger.info(f"Model weights: {self.weights}")

        return self

    def _fit_meta_learner_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: List[str],
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace inverse-MAE-on-train weighting (B5) with a constrained
        meta-learner fit to OOF positions vs. ideal positions.

        Phase 2.1: both forecast and policy members produce OOF via
        PurgedWalkForward (k refits each). Policy refits need the original
        fit_kwargs so we can re-pass features and per-fold sliced
        X_train_with_close.
        """
        if self.target_column not in X.columns:
            logger.warning(
                f"Meta-learner needs '{self.target_column}' in X to compute realized vol; "
                "skipping dynamic re-weighting."
            )
            return

        oof = self._compute_oof_positions(X, y, model_names, fit_kwargs or {})
        if not oof:
            logger.warning("No OOF positions computed; keeping config weights.")
            return

        idx = next(iter(oof.values())).index
        members = list(oof.keys())
        for name in members[1:]:
            idx = idx.intersection(oof[name].index)
        if len(idx) < 5:
            logger.warning(f"Only {len(idx)} overlapping OOF rows; keeping config weights.")
            return

        close = X.loc[idx, self.target_column]
        sigma = realized_vol(close, window=self.vol_window, vol_floor=self.vol_floor)
        realized_ret = close.pct_change(self.horizon).shift(-self.horizon).to_numpy()
        y_ideal = ideal_position(
            np.nan_to_num(realized_ret, nan=0.0),
            sigma,
            target_vol=self.target_vol,
            cap=self.position_cap,
        )

        valid = np.isfinite(realized_ret)
        if valid.sum() < 5:
            logger.warning("Too few finite realized-return rows for meta-learner; keeping config weights.")
            return

        P = np.column_stack([oof[name].loc[idx].to_numpy() for name in members])[valid]
        target = y_ideal[valid]

        w0 = np.full(len(members), 1.0 / len(members))
        result = minimize(
            lambda w: float(np.mean((target - P @ w) ** 2)),
            w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(members),
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"ftol": 1e-9, "maxiter": 200},
        )
        if not result.success:
            logger.warning(f"Meta-learner SLSQP did not converge ({result.message}); keeping config weights.")
            return

        for name, w in zip(members, result.x):
            self.weights[name] = float(w)
        self.errors = {
            name: float(np.mean(np.abs(target - oof[name].loc[idx].to_numpy()[valid])))
            for name in members
        }
        logger.info(f"Meta-learner weights: {self.weights}")

        # Phase 2.5: fit position-level conformal calibrator on the same OOF
        # residuals the meta-learner just regressed against. The blended OOF
        # uses the freshly-fit weights so the calibration matches inference
        # behavior of predict().
        try:
            w_arr = np.array([self.weights[n] for n in members], dtype=np.float64)
            blended_oof = P @ w_arr
            blended_oof = np.clip(blended_oof, -self.position_cap, self.position_cap)
            self.conformal = EnbPICalibrator().fit(
                oof_predictions=blended_oof,
                targets=target,
                position_cap=self.position_cap,
            )
            self.aci = ACIState(
                alpha_target=self.conformal_alpha_target,
                gamma=self.conformal_aci_gamma,
                alpha_t=self.conformal_alpha_target,
            )
            logger.info(
                f"Conformal calibrator fit on {len(target)} OOF residuals "
                f"(alpha_target={self.conformal_alpha_target}, gamma={self.conformal_aci_gamma})."
            )
        except ValueError as e:
            logger.warning(f"Conformal calibration skipped ({e}); predict_band will fall back to max bands.")
            self.conformal = None
            self.aci = None

    def _compute_oof_positions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: List[str],
        fit_kwargs: Dict[str, Any],
    ) -> Dict[str, pd.Series]:
        """Produce per-model OOF position vectors aligned to X.index.

        Phase 2.1: every member is OOF'd via PurgedWalkForward. Forecast
        members refit cheaply per fold; policy members refit too (the old
        train-tail-holdout shortcut was in-sample-leaky because the same
        model was already trained on those rows). Caller is on the hook
        for the n_splits × RL-fit cost — keep `meta_cv_folds` modest.
        """
        oof: Dict[str, pd.Series] = {}
        if self.target_column not in X.columns:
            return oof

        try:
            splitter = PurgedWalkForward(
                n_splits=self.meta_cv_folds,
                purge_horizon=self.horizon,
                embargo_pct=self.meta_embargo_pct,
                expanding=True,
            )
            folds = list(splitter.split(X))
        except ValueError as e:
            logger.warning(f"PurgedWalkForward setup failed ({e}); skipping OOF.")
            return oof
        if not folds:
            logger.warning("PurgedWalkForward yielded no folds; skipping OOF.")
            return oof

        close_arr = X[self.target_column].to_numpy()
        sigma_full = realized_vol(
            X[self.target_column],
            window=self.vol_window,
            vol_floor=self.vol_floor,
        )

        for name in model_names:
            if name not in self.models:
                continue
            try:
                if is_forecast(name):
                    oof[name] = self._oof_forecast(
                        name, X, y, close_arr, sigma_full, folds
                    )
                elif is_policy(name):
                    oof[name] = self._oof_policy(name, X, y, fit_kwargs, folds)
            except Exception as e:
                logger.error(
                    f"OOF generation failed for {name}: {e}", exc_info=True
                )
        return oof

    def _oof_forecast(
        self,
        name: str,
        X: pd.DataFrame,
        y: pd.Series,
        close_arr: np.ndarray,
        sigma_full: np.ndarray,
        folds: List[tuple],
    ) -> pd.Series:
        out = np.full(len(X), np.nan)
        for tr_idx, va_idx in folds:
            fold_model = self._create_model(name)
            if fold_model is None:
                continue
            fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = fold_model.predict(X.iloc[va_idx])
            if len(preds) != len(va_idx):
                continue
            out[va_idx] = forecast_to_position(
                np.asarray(preds, dtype=np.float64),
                close_arr[va_idx],
                sigma_full[va_idx],
                target_vol=self.target_vol,
                cap=self.position_cap,
            )
        return pd.Series(out, index=X.index).dropna()

    def _oof_policy(
        self,
        name: str,
        X: pd.DataFrame,
        y: pd.Series,
        fit_kwargs: Dict[str, Any],
        folds: List[tuple],
    ) -> pd.Series:
        """Per-fold policy refit + predict on test slice. Assumes
        `{name}_X_train_with_close` (the unscaled OHLCV+features frame the
        policy needs for its env) is row-aligned with X — same assumption
        the top-level fit() makes when it passes the full frame through.
        """
        x_full = fit_kwargs.get(f"{name}_X_train_with_close")
        if x_full is None:
            logger.warning(
                f"Policy OOF for {name} skipped: missing "
                f"'{name}_X_train_with_close' kwarg."
            )
            return pd.Series(dtype=np.float64)
        if len(x_full) != len(X):
            logger.warning(
                f"Policy OOF for {name} skipped: "
                f"'{name}_X_train_with_close' has {len(x_full)} rows, "
                f"X has {len(X)} — cannot iloc-slice consistently."
            )
            return pd.Series(dtype=np.float64)

        out = np.full(len(X), np.nan)
        for tr_idx, va_idx in folds:
            fold_model = self._create_model(name)
            if fold_model is None:
                continue
            ok = self._fit_policy_member(
                name=name,
                model=fold_model,
                X_with_close=x_full.iloc[tr_idx],
                y=y.iloc[tr_idx],
                kwargs=fit_kwargs,
                persist=False,
            )
            if not ok:
                continue
            preds = fold_model.predict(X.iloc[va_idx])
            if len(preds) != len(va_idx):
                logger.warning(
                    f"Policy {name} OOF fold returned {len(preds)} preds "
                    f"for {len(va_idx)} rows; skipping fold."
                )
                continue
            out[va_idx] = np.clip(
                np.asarray(preds, dtype=np.float64),
                -self.position_cap,
                self.position_cap,
            )
        return pd.Series(out, index=X.index).dropna()

    def predict(
        self,
        X: pd.DataFrame,
        *,
        policy_X: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Return ensemble positions in [-1, 1].

        Forecast members' ŷ are mapped to positions via inverse-volatility
        sizing against X[target_column]; policy members' outputs pass through
        with a clip. The weighted blend lives in position space, not the
        mixed price/position space that produced B8.
        """
        if not self.is_fitted or not self.models:
            logger.warning("Ensemble model not fitted yet")
            return np.array([])

        positions = self._predict_positions_per_model(X, policy_X=policy_X)
        if not positions:
            logger.warning("No valid model positions available")
            return np.array([])

        blended = np.zeros(len(X))
        total_weight = 0.0
        for name, pos in positions.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue
            blended += w * pos
            total_weight += w

        if total_weight == 0:
            logger.warning("Total weight zero; falling back to equal-weight position average.")
            blended = np.mean(np.stack(list(positions.values()), axis=0), axis=0)
        elif not np.isclose(total_weight, 1.0):
            blended /= total_weight

        return np.clip(blended, -self.position_cap, self.position_cap)

    def has_conformal(self) -> bool:
        """True when fit() produced a usable conformal calibrator + ACI state."""
        return self.conformal is not None and self.conformal.is_fitted and self.aci is not None

    def predict_band(
        self,
        X: pd.DataFrame,
        *,
        policy_X: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Return ``(lower, point, upper)`` position-space conformal bands.

        Falls back to maximally-wide bands (±position_cap) when no conformal
        calibrator was fit — this lets downstream code call predict_band
        unconditionally and still get a no-op interval.
        """
        point = self.predict(X, policy_X=policy_X)
        if not self.has_conformal() or len(point) == 0:
            lower = np.full(len(point), -self.position_cap)
            upper = np.full(len(point), self.position_cap)
            return lower, point, upper
        alpha = self.aci.current_alpha()  # type: ignore[union-attr]
        lower, upper = self.conformal.band(point, alpha=alpha)  # type: ignore[union-attr]
        return lower, point, upper

    def update_aci(self, in_band: bool) -> Optional[float]:
        """Apply one ACI online update against an observed coverage event.

        Returns the new alpha_t, or None when no conformal state exists.
        Caller is responsible for evaluating whether the realized outcome
        landed inside the previously-emitted band (the ensemble does not
        track forward-horizon outcomes itself).
        """
        if not self.has_conformal():
            return None
        return self.aci.update(in_band)  # type: ignore[union-attr]

    def _predict_positions_per_model(
        self,
        X: pd.DataFrame,
        *,
        policy_X: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Run each member's predict and convert to a position vector."""
        if self.target_column not in X.columns:
            logger.error(
                f"Ensemble.predict requires '{self.target_column}' column in X "
                "(needed for forecast→position vol sizing)."
            )
            return {}
        if policy_X is not None and len(policy_X) != len(X):
            raise ValueError(
                f"policy_X length {len(policy_X)} does not match forecast X length {len(X)}"
            )
        close = X[self.target_column].to_numpy()
        sigma = realized_vol(X[self.target_column], window=self.vol_window, vol_floor=self.vol_floor)

        positions: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            try:
                kind = self.kinds.get(name, model_kind(name))
                member_X = policy_X if kind == "policy" and policy_X is not None else X
                raw = model.predict(member_X)
                if len(raw) != len(member_X):
                    if len(raw) > 0:
                        logger.warning(
                            f"{name}: predict returned {len(raw)} rows for {len(member_X)} input rows; skipping."
                        )
                    continue
                raw_arr = np.asarray(raw, dtype=np.float64)
                if kind == "forecast":
                    positions[name] = forecast_to_position(
                        raw_arr, close, sigma, target_vol=self.target_vol, cap=self.position_cap
                    )
                else:
                    positions[name] = np.clip(raw_arr, -self.position_cap, self.position_cap)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        return positions

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble positions against the ideal (perfect-foresight) bet.

        The ensemble emits positions; the apples-to-apples target is the
        ideal_position derived from realized return / realized vol. Forecast
        members are scored on price MSE too (the quantity they actually
        regress against); policy members are scored only in position space.
        """
        if not self.is_fitted:
            logger.warning("Ensemble model not fitted yet")
            return {}
        if self.target_column not in X_test.columns:
            logger.warning(
                f"evaluate needs '{self.target_column}' in X_test for ideal-position computation."
            )
            return {}

        try:
            y_pred = self.predict(X_test)
            if len(y_pred) == 0 or len(y_pred) != len(y_test):
                logger.warning("Prediction length mismatch or no predictions available for evaluation")
                return {}

            close = X_test[self.target_column].to_numpy()
            sigma = realized_vol(X_test[self.target_column], window=self.vol_window, vol_floor=self.vol_floor)
            y_test_arr = np.asarray(y_test, dtype=np.float64)
            realized_ret = (y_test_arr - close) / close
            y_ideal = ideal_position(realized_ret, sigma, target_vol=self.target_vol, cap=self.position_cap)

            # A forecast member can emit a NaN prediction on real data (e.g.
            # ARIMA on a degenerate window); it propagates through the blend
            # into y_pred. sklearn's metrics reject any NaN, which previously
            # made the whole evaluate() return {} ("Input contains NaN") and
            # dropped the fold's metrics. Score only the finite rows.
            y_pred = np.asarray(y_pred, dtype=np.float64)
            finite = np.isfinite(y_pred) & np.isfinite(y_ideal)
            if finite.sum() < 2:
                logger.warning(
                    "evaluate: fewer than 2 finite (y_ideal, y_pred) rows; "
                    "skipping ensemble metrics."
                )
                return {}
            yi, yp = y_ideal[finite], y_pred[finite]
            metrics = {
                "position_mae": mean_absolute_error(yi, yp),
                "position_mse": mean_squared_error(yi, yp),
                "position_rmse": float(np.sqrt(mean_squared_error(yi, yp))),
                "position_r2": r2_score(yi, yp),
                "position_n_finite": int(finite.sum()),
            }

            for name, model in self.models.items():
                try:
                    model_preds = np.asarray(model.predict(X_test), dtype=np.float64)
                    if len(model_preds) != len(y_test):
                        if len(model_preds) > 0:
                            logger.warning(f"Individual eval skipped for {name}: length mismatch.")
                        continue
                    if self.kinds.get(name) == "forecast":
                        pf = np.isfinite(y_test_arr) & np.isfinite(model_preds)
                        if pf.sum() >= 2:
                            metrics[f"{name}_price_mae"] = mean_absolute_error(
                                y_test_arr[pf], model_preds[pf]
                            )
                            metrics[f"{name}_price_rmse"] = float(
                                np.sqrt(mean_squared_error(y_test_arr[pf], model_preds[pf]))
                            )
                        pos = forecast_to_position(
                            model_preds, close, sigma, target_vol=self.target_vol, cap=self.position_cap
                        )
                    else:
                        pos = np.clip(model_preds, -self.position_cap, self.position_cap)
                    posf = np.isfinite(y_ideal) & np.isfinite(pos)
                    if posf.sum() >= 2:
                        metrics[f"{name}_position_mae"] = mean_absolute_error(
                            y_ideal[posf], pos[posf]
                        )
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {}

    def get_model_contributions(
        self,
        X: pd.DataFrame,
        *,
        policy_X: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Per-model positions, weights, and weighted contributions.

        Columns: `{name}_position`, `{name}_weight`, `{name}_contrib`, plus
        `ensemble_position`. Forecast outputs are vol-mapped before display
        so all `_position` columns are in the same [-1, 1] space.
        """
        if not self.is_fitted:
            logger.warning("Ensemble model not fitted yet")
            return pd.DataFrame()

        ensemble_pos = self.predict(X, policy_X=policy_X)
        if len(ensemble_pos) != len(X):
            logger.warning("Ensemble prediction length mismatch in get_model_contributions")
            return pd.DataFrame()

        positions = self._predict_positions_per_model(X, policy_X=policy_X)
        if not positions:
            return pd.DataFrame()

        data: Dict[str, np.ndarray] = {}
        for name, pos in positions.items():
            w = float(self.weights.get(name, 0.0))
            data[f"{name}_position"] = pos
            data[f"{name}_weight"] = np.full(len(X), w)
            data[f"{name}_contrib"] = pos * w
        df = pd.DataFrame(data, index=X.index)
        df["ensemble_position"] = ensemble_pos
        return df

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save the ensemble model state (config, weights, errors) to disk.

        Sub-models like LSTM, XGBoost are assumed to be saved by their own methods
        if needed, or handled separately (like LSTMPPO's path being stored).

        Args:
            directory: Base directory to save the ensemble state file.
                     The actual file will be named within this directory.

        Returns:
            Path to the saved *directory* containing the ensemble state.
        """
        # Ensure the target directory exists
        directory.mkdir(exist_ok=True, parents=True)

        # Define the path for the ensemble state file within the directory
        model_state_path = directory / f"ensemble_state_h{self.horizon}.pkl"

        # Create a serializable copy of the model state
        save_state: Dict[str, Any] = {
            "target_column": self.target_column,
            "horizon": self.horizon,
            "config": self.config,
            "weights": self.weights,
            "errors": self.errors,
            "is_fitted": self.is_fitted,
            "kinds": self.kinds,
            "vol_window": self.vol_window,
            "vol_floor": self.vol_floor,
            "target_vol": self.target_vol,
            "position_cap": self.position_cap,
            "meta_cv_folds": self.meta_cv_folds,
            "meta_embargo_pct": self.meta_embargo_pct,
            "conformal_alpha_target": self.conformal_alpha_target,
            "conformal_aci_gamma": self.conformal_aci_gamma,
            "conformal_abs_residuals": (
                self.conformal.abs_residuals if self.conformal is not None else None
            ),
            "conformal_position_cap": (
                self.conformal.position_cap if self.conformal is not None else None
            ),
            "aci_alpha_t": self.aci.alpha_t if self.aci is not None else None,
            "models": {},
        }

        # Include fitted models state/path
        for name in self.models.keys():
            if name == "lstm_ppo":
                if self.lstm_ppo_save_path:
                    save_state["models"][name] = str(self.lstm_ppo_save_path)
                else:
                    logger.warning("LSTMPPO model was fitted but no save path was stored.")
            elif name == "xlstm_ppo":
                if self.xlstm_ppo_save_path:
                    save_state["models"][name] = str(self.xlstm_ppo_save_path)
                else:
                    logger.warning("XLSTMPPOAgent model was fitted but no save path was stored.")
            elif name == "xlstm_grpo":
                if self.xlstm_grpo_save_path:
                    save_state["models"][name] = str(self.xlstm_grpo_save_path)
                else:
                    logger.warning("XLSTMGRPOAgent model was fitted but no save path was stored.")
            else:
                # Forecast members (arima/prophet/lstm/xgboost) own their
                # save(directory)/load(path); delegate to them and store the
                # returned path RELATIVE to this ensemble dir so the run stays
                # relocatable. Previously this branch was a no-op, so a loaded
                # ensemble silently came back with UNFITTED members → all-zero
                # positions → zero trades in every backtest fold.
                member = self.models[name]
                if not getattr(member, "is_fitted", False):
                    logger.warning(
                        f"Member '{name}' not fitted at save time; skipping persistence."
                    )
                    continue
                try:
                    saved_path = Path(member.save(directory))
                    rel = saved_path.resolve().relative_to(directory.resolve())
                    save_state["models"][name] = {"forecast_rel_path": str(rel)}
                except Exception as e:
                    logger.error(f"Error saving member '{name}': {e}")

        # Save the state to the pickle file
        try:
            with open(model_state_path, "wb") as f:
                pickle.dump(save_state, f)
            logger.info(f"Ensemble model state saved to {model_state_path}")
            return directory  # Return the directory path
        except Exception as e:
            logger.error(f"Error saving ensemble state: {e}")
            return directory  # Still return directory even on error?

    def load(self, model_path: Path) -> "EnsembleModel":
        """Load model state from disk.

        Args:
            model_path: Path to the saved model state file

        Returns:
            Loaded model instance (self)
        """
        with open(model_path, "rb") as f:
            loaded_state = pickle.load(f)

        # Restore attributes
        self.target_column = loaded_state["target_column"]
        self.horizon = loaded_state["horizon"]
        self.config = loaded_state["config"]
        self.weights = loaded_state["weights"]
        self.errors = loaded_state["errors"]
        self.is_fitted = loaded_state["is_fitted"]
        self.kinds = loaded_state.get("kinds", {n: model_kind(n) for n in self.weights})
        self.vol_window = loaded_state.get("vol_window", 20)
        self.vol_floor = loaded_state.get("vol_floor", 5e-3)
        self.target_vol = loaded_state.get("target_vol", 1.0)
        self.position_cap = loaded_state.get("position_cap", 1.0)
        self.meta_cv_folds = loaded_state.get("meta_cv_folds", 3)
        self.meta_embargo_pct = loaded_state.get("meta_embargo_pct", 0.01)

        # Phase 2.5: restore conformal calibrator + ACI state when present.
        self.conformal_alpha_target = loaded_state.get("conformal_alpha_target", 0.10)
        self.conformal_aci_gamma = loaded_state.get("conformal_aci_gamma", 0.01)
        residuals = loaded_state.get("conformal_abs_residuals")
        if residuals is not None and len(residuals) > 0:
            cal = EnbPICalibrator()
            cal.abs_residuals = np.asarray(residuals, dtype=np.float64)
            cal.position_cap = float(
                loaded_state.get("conformal_position_cap", 1.0)
            )
            cal.is_fitted = True
            self.conformal = cal
            self.aci = ACIState(
                alpha_target=self.conformal_alpha_target,
                gamma=self.conformal_aci_gamma,
                alpha_t=float(
                    loaded_state.get("aci_alpha_t", self.conformal_alpha_target)
                ),
            )
        else:
            self.conformal = None
            self.aci = None

        self.models = {}  # Initialize empty

        # Initialize path attributes
        self.lstm_ppo_save_path = None
        self.xlstm_ppo_save_path = None
        self.xlstm_grpo_save_path = None

        # Load models
        for name, saved_model_info in loaded_state["models"].items():
            if name == "lstm_ppo":
                # Load LSTMPPO from its saved path string
                try:
                    lstm_ppo_path_str = str(saved_model_info)
                    lstm_ppo_path = Path(lstm_ppo_path_str)
                    self.lstm_ppo_save_path = lstm_ppo_path  # Store the loaded path

                    if lstm_ppo_path.exists() and (lstm_ppo_path / "model.zip").exists():
                        # Re-initialize the model instance before loading
                        lstm_ppo_model = LSTMPPO(target_column=self.target_column, horizon=self.horizon)
                        lstm_ppo_model.load(lstm_ppo_path / "model.zip")
                        self.models[name] = lstm_ppo_model
                    else:
                        logger.warning(f"LSTMPPO model path or file not found: {lstm_ppo_path}")
                        self.is_fitted = False  # Mark as not fitted if LSTMPPO failed to load
                except Exception as e:
                    logger.error(f"Error loading LSTMPPO model from path {saved_model_info}: {e}")
                    self.is_fitted = False  # Mark as not fitted if LSTMPPO failed to load
            elif name == "xlstm_ppo":
                # Load XLSTMPPOAgent from its saved path string
                try:
                    from src.models.xlstm_ppo import XLSTMPPOAgent

                    xlstm_ppo_path_str = str(saved_model_info)
                    xlstm_ppo_path = Path(xlstm_ppo_path_str)
                    self.xlstm_ppo_save_path = xlstm_ppo_path  # Store the loaded path

                    if xlstm_ppo_path.exists() and (xlstm_ppo_path / "config.pkl").exists():
                        # Re-initialize the model instance before loading
                        xlstm_ppo_model = XLSTMPPOAgent(target_column=self.target_column, horizon=self.horizon)
                        xlstm_ppo_model.load(xlstm_ppo_path)
                        self.models[name] = xlstm_ppo_model
                    else:
                        logger.warning(f"XLSTMPPOAgent model path or file not found: {xlstm_ppo_path}")
                        self.is_fitted = False  # Mark as not fitted if XLSTMPPOAgent failed to load
                except Exception as e:
                    logger.error(f"Error loading XLSTMPPOAgent model from path {saved_model_info}: {e}")
                    self.is_fitted = False  # Mark as not fitted if XLSTMPPOAgent failed to load
            elif name == "xlstm_grpo":
                # Load XLSTMGRPOAgent from its saved path string
                try:
                    from src.models.xlstm_grpo import XLSTMGRPOAgent

                    xlstm_grpo_path_str = str(saved_model_info)
                    xlstm_grpo_path = Path(xlstm_grpo_path_str)
                    self.xlstm_grpo_save_path = xlstm_grpo_path  # Store the loaded path

                    if xlstm_grpo_path.exists() and (xlstm_grpo_path / "config.pkl").exists():
                        # Re-initialize the model instance before loading
                        xlstm_grpo_model = XLSTMGRPOAgent(target_column=self.target_column, horizon=self.horizon)
                        xlstm_grpo_model.load(xlstm_grpo_path)
                        self.models[name] = xlstm_grpo_model
                    else:
                        logger.warning(f"XLSTMGRPOAgent model path or file not found: {xlstm_grpo_path}")
                        self.is_fitted = False  # Mark as not fitted if XLSTMGRPOAgent failed to load
                except Exception as e:
                    logger.error(f"Error loading XLSTMGRPOAgent model from path {saved_model_info}: {e}")
                    self.is_fitted = False  # Mark as not fitted if XLSTMGRPOAgent failed to load
            else:
                # Forecast members: reconstruct the instance and load it from
                # the path stored (relative to this ensemble dir) by save().
                # Legacy pickles that stored the object directly still load via
                # the fallback branch.
                if isinstance(saved_model_info, dict) and "forecast_rel_path" in saved_model_info:
                    member = self._create_model(name)
                    if member is None:
                        logger.warning(
                            f"Could not re-create member '{name}'; marking ensemble unfitted."
                        )
                        self.is_fitted = False
                        continue
                    member_path = Path(model_path).parent / saved_model_info["forecast_rel_path"]
                    try:
                        member.load(member_path)
                        if not getattr(member, "is_fitted", False):
                            logger.warning(
                                f"Member '{name}' loaded from {member_path} but reports not fitted."
                            )
                            self.is_fitted = False
                        self.models[name] = member
                    except Exception as e:
                        logger.error(
                            f"Error loading member '{name}' from {member_path}: {e}"
                        )
                        self.is_fitted = False
                else:
                    # Legacy: a directly-pickled model object.
                    self.models[name] = saved_model_info

        # Re-create models specified in config but not found in loaded state (optional, depends on desired behavior)
        for model_config in self.config.models:
            if model_config.enabled and model_config.name not in self.models:
                logger.warning(
                    f"Model '{model_config.name}' defined in config but not found in saved state. "
                    f"Attempting to re-initialize."
                )
                model_instance = self._create_model(model_config.name)
                if model_instance:
                    self.models[model_config.name] = model_instance
                    # Model will need refitting if loaded this way
                    self.is_fitted = False  # Mark ensemble as not fully fitted if we had to re-initialize

        logger.info(f"Ensemble model state loaded from {model_path}")

        return self
