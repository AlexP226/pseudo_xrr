# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:30:24 2026
@author: shenc

functions related to the bulk background fitting
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def bulkbkg_model(x, y0, F, t):
    """
    bulk bkg: offset exponential model

    Parameters
    ----------
    x : numpy array, Q

    Returns
    -------
    numpy array
        y0 + F * np.exp(x / t).

    """
    return y0 + F * np.exp(x / t)

def bulkbkg_is_nearly_constant(y, rtol=1e-3):
    """
    function to determine if the bkg is nearly a constant

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    rtol : TYPE, optional
        DESCRIPTION. The default is 1e-3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mu = float(np.mean(y))
    return float(np.std(y)) < rtol * max(abs(mu), 1.0)

def bulkbkg_fit(
    x, y,
    *,
    # positivity + optional physics bounds
    y0_bounds=(1e-12, np.inf),
    F_bounds=(1e-12, np.inf),
    t_bounds=(1e-12, np.inf),
    # prior for y0 initial guess (only affects start point, not the final fit unless you also bound y0)
    y0_clip_for_init=(200.0, 2000.0),
    nearly_constant_rtol=1e-3,
    allow_fit_even_if_flat=True,
    # prevent runaway t (set None to disable)
    t_upper_multiple_of_span=200.0,
    maxfev=200000,
    # plot fit
    plot_fit=False,
    ):
    """
    Fit y = y0 + F * exp(x/t) with y0,F,t > 0 (by default).

    Returns a dict with:
      ok (bool), popt ([y0,F,t]), pcov, x, y, y_fit, flat (bool), message (str)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError(f"x and y must have same length; got {x.size} and {y.size}")
    if x.size < 3:
        raise ValueError("Need at least 3 points to fit.")

    # sort by x (helps stability; doesn't change the fit)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # flat-ish detection
    flat = bulkbkg_is_nearly_constant(y, rtol=nearly_constant_rtol)
    if flat and not allow_fit_even_if_flat:
        y0 = float(np.mean(y))
        popt = np.array([y0, 0.0, np.nan], dtype=float)
        return {
            "ok": True,
            "flat": True,
            "popt": popt,
            "pcov": np.full((3, 3), np.inf),
            "x": x,
            "y": y,
            "y_fit": np.full_like(y, y0, dtype=float),
            "message": "Nearly constant data: returned y0=mean(y), F=0, t=nan",
        }

    # shift x for numerical stability, then map F back (keeps same y0, t; adjusts F)
    x0 = float(np.min(x))
    xs = x - x0
    span = float(np.max(xs) - np.min(xs))

    # initial guesses
    y0_guess = float(np.clip(np.percentile(y, 10), y0_clip_for_init[0], y0_clip_for_init[1]))
    F_guess  = float(max(np.max(y) - y0_guess, 1e-12))
    t_guess  = float(max(span / 5.0, 1e-12))
    p0 = (y0_guess, F_guess, t_guess)

    # bounds (in shifted-x parameterization)
    lo = (y0_bounds[0], F_bounds[0], t_bounds[0])
    hi = (y0_bounds[1], F_bounds[1], t_bounds[1])

    # optional cap on t to avoid runaway in weakly-informative/flat-ish cases
    if t_upper_multiple_of_span is not None and np.isfinite(span) and span > 0:
        hi = (hi[0], hi[1], min(hi[2], t_upper_multiple_of_span * span))

    try:
        popt_s, pcov = curve_fit(
            bulkbkg_model, xs, y,
            p0=p0,
            bounds=(lo, hi),
            maxfev=maxfev
        )

        # Map back to original x:
        # y = y0 + F_s*exp((x-x0)/t) = y0 + (F_s*exp(-x0/t))*exp(x/t)
        y0, F_s, t = [float(v) for v in popt_s]
        F = float(F_s * np.exp(-x0 / t))
        popt = np.array([y0, F, t], dtype=float)

        y_fit = bulkbkg_model(x, *popt)

        # If you want to see if the fit is "stable-ish", these flags help:
        msg = "Fit succeeded"
        if flat:
            msg += " (data flagged nearly-constant)"
        if span > 0 and np.isfinite(t) and t > 50.0 * span:
            msg += " (warning: t very large vs x-span)"
        if np.isfinite(F) and F < 1e-3 * max(abs(np.mean(y)), 1.0):
            msg += " (warning: F ~ 0)"

        return {
            "ok": True,
            "flat": flat,
            "popt": popt,
            "pcov": pcov,
            "x": x,
            "y": y,
            "y_fit": y_fit,
            "message": msg,
        }

    except Exception as e:
        return {
            "ok": False,
            "flat": flat,
            "popt": np.array([np.nan, np.nan, np.nan], dtype=float),
            "pcov": np.full((3, 3), np.nan),
            "x": x,
            "y": y,
            "y_fit": np.full_like(y, np.nan, dtype=float),
            "message": f"Fit failed: {type(e).__name__}: {e}",
        }

def bulkbkg_plot_fit(result, *, ax=None):
    """
    Optional helper: plot data + fit for the output dict of fit_exp_offset_xy.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.plot(result["x"], result["y"], ".", label="data")

    if result["ok"]:
        ax.plot(result["x"], result["y_fit"], "-", linewidth=2, label="fit")
        y0, F, t = result["popt"]
        ax.set_title(f"y0={y0:.4g}, F={F:.4g}, t={t:.4g}")
    else:
        ax.set_title(result["message"])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax    

def bulkbkg_predict(Q, params):
    """
    Predict bulkbkg intensity from params = [y0, F, t].
    If t is NaN/inf or F is ~0, returns y ~ y0 (flat-line fallback).
    Q is a 2D array, and will return a 2D array
    """
    x_new = np.asarray(Q, dtype=float)
    y0, F, t = [float(v) for v in params]

    # Flat/degenerate fallback
    if (not np.isfinite(t)) or (not np.isfinite(F)) or (abs(F) < 1e-15):
        return np.full_like(x_new, y0, dtype=float)

    return y0 + F * np.exp(x_new / t)

