"""Interactive Arrhenius fit to extract E_a and k0 from cure-time data."""

import math
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Arrhenius Kinetics Fit", layout="wide")


def _init_points() -> List[dict]:
    """Provide a small, realistic seed dataset for ln(t) vs 1/T fitting."""
    # Derived from an illustrative Arrhenius pair (E_a ≈ 120 kJ/mol, k0 ≈ 1e9 1/s)
    # converted to seconds at representative cure temperatures.
    seeds = [
        {"Temperature (°C)": 380.0, "Time (s)": 3.95},
        {"Temperature (°C)": 400.0, "Time (s)": 2.05},
        {"Temperature (°C)": 420.0, "Time (s)": 1.10},
        {"Temperature (°C)": 440.0, "Time (s)": 0.62},
        {"Temperature (°C)": 460.0, "Time (s)": 0.35},
    ]
    return seeds


st.title("Arrhenius best-fit for cure kinetics")

st.markdown(
    r"""
Use your own cure-time measurements to extract activation energy and the pre-exponential
factor by fitting the linearized Arrhenius form:

\[ \ln t = \frac{E_a}{R} \frac{1}{T} - \ln k_0 + \ln C_{\text{target}} \]

* Slope → \(E_a / R\); intercept → \(-\ln k_0 + \ln C_{\text{target}}\)
* Enter at least five points (temperature in °C, time in seconds) to solve.
    """
)

with st.expander("Fitting inputs and target assumptions", expanded=True):
    c1, c2, c3 = st.columns(3)
    R_gas = c1.number_input(
        "Gas constant R [J/mol·K]",
        value=8.314,
        min_value=1e-6,
        step=0.001,
        help="Editable so you can match unit systems; defaults to J/mol·K.",
    )
    C_target = c2.number_input(
        "Cure index target (dimensionless)",
        value=1.0,
        min_value=1e-6,
        step=0.1,
        help="Target cure index used in the intercept term (ln C_target).",
    )
    temp_limit = c3.slider(
        "Cure temperature guidance (°C)",
        min_value=200.0,
        max_value=400.0,
        value=(200.0, 400.0),
        help="Reference range only; data outside the band still fit."
    )

if "arrhenius_points" not in st.session_state:
    st.session_state.arrhenius_points = _init_points()

st.subheader("Enter temperature / time data")
st.caption(
    "Add or delete rows as needed. Provide temperature in °C and cure time in seconds;"
    " the fit updates live when five or more valid rows are present."
)

edited = st.data_editor(
    pd.DataFrame(st.session_state.arrhenius_points),
    num_rows="dynamic",
    use_container_width=True,
    key="arrhenius_editor",
    column_config={
        "Temperature (°C)": st.column_config.NumberColumn(
            "Temperature (°C)", min_value=1.0, format="%.2f"
        ),
        "Time (s)": st.column_config.NumberColumn(
            "Time (s)", min_value=1e-6, format="%.4f"
        ),
    },
)

# Persist edits for later reruns
st.session_state.arrhenius_points = edited.to_dict("records")

clean_df = edited.dropna()
clean_df = clean_df[(clean_df["Temperature (°C)"] > 0) & (clean_df["Time (s)"] > 0)]

if len(clean_df) < 5:
    st.info("Add at least five valid points to compute an Arrhenius fit.")
    st.stop()

temp_K = clean_df["Temperature (°C)"] + 273.15
inv_T = 1.0 / temp_K
ln_t = np.log(clean_df["Time (s)"])

# Linear regression (first-order polynomial) on (1/T, ln t)
slope, intercept = np.polyfit(inv_T, ln_t, 1)
Ea = slope * R_gas
k0 = math.exp(-intercept + math.log(C_target))

fit_line_x = np.linspace(inv_T.min(), inv_T.max(), 200)
fit_line_y = slope * fit_line_x + intercept

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_scatter(
        x=inv_T,
        y=ln_t,
        mode="markers",
        name="Data",
        marker=dict(color="#1f77b4", size=9),
    )
    fig.add_scatter(
        x=fit_line_x,
        y=fit_line_y,
        mode="lines",
        name="Linear fit",
        line=dict(color="#ff7f0e"),
    )
    fig.update_layout(
        xaxis_title="1 / T [1/K]",
        yaxis_title="ln t",
        legend_title="Series",
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Fit results")
    st.metric("Slope (E_a / R)", f"{slope:0.3e} [K]")
    st.metric("Activation energy E_a", f"{Ea/1000:0.2f} kJ/mol")
    st.metric("Pre-exponential k₀", f"{k0:0.3e} 1/s")
    st.caption(
        "Fit derived from ln t vs 1/T. Defaults are illustrative; adjust R and "
        "C_target to match your cure definition."
    )

st.divider()
st.markdown(
    r"""
**How the fit works**

1. Convert each entry to \(1/T\) (K⁻¹) and \(\ln t\).
2. Perform a simple linear regression \(\ln t = m (1/T) + b\).
3. Recover kinetics from \(m = E_a/R\) and \(b = -\ln k_0 + \ln C_{\text{target}}\).

If you have a vendor cure-time chart, digitize the points here to back-calculate
an effective \(E_a\) and \(k_0\) for your enamel system.
    """
)
