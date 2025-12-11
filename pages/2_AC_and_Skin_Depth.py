import math
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

MU0 = 4 * math.pi * 1e-7

st.title("AC Effects & Skin Depth")

st.caption(
    "Permeability constant μ₀ = 4π × 10⁻⁷ H/m (Engineering Toolbox: "
    "https://www.engineeringtoolbox.com/permeability-d_1923.html)."
)

st.markdown(
    """
    Estimate when skin effect starts to penalize magnet wire and derive the skin depth for
    your frequency, resistivity, and permeability. Leave one field blank to solve for it.
    """
)

st.header("Skin Depth")
st.latex(r"\delta = \sqrt{\frac{2\rho}{\omega \mu}} = \sqrt{\frac{\rho}{\pi f \mu_0 \mu_r}}")

col1, col2, col3 = st.columns(3)
with col1:
    rho = st.number_input("Resistivity ρ [Ω·m]", value=1.724e-8, format="%e")
with col2:
    freq = st.number_input("Frequency f [Hz] (blank to solve)", value=0.0, format="%f")
    freq_val = freq if freq > 0 else None
with col3:
    mu_r = st.number_input("Relative permeability μᵣ (blank to solve)", value=1.0, format="%f")
    mu_r_val = mu_r if mu_r > 0 else None

skin_depth = st.number_input("Skin depth δ [m] (blank to solve)", value=0.0, format="%e")
skin_val = skin_depth if skin_depth > 0 else None

rho_sym, f_sym, mu_r_sym, delta_sym = sp.symbols("rho f mu_r delta", positive=True)
equation = sp.Eq(delta_sym, sp.sqrt(rho_sym / (sp.pi * f_sym * MU0 * mu_r_sym)))
symbol_values = {rho_sym: rho, f_sym: freq_val, mu_r_sym: mu_r_val, delta_sym: skin_val}
missing_symbol, solved_value = solve_missing_symbol(equation, symbol_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == delta_sym:
        st.success(f"δ ≈ {solved_value*1e3:.3f} mm")
    elif missing_symbol == f_sym:
        st.success(f"Frequency ≈ {solved_value:.1f} Hz")
    elif missing_symbol == mu_r_sym:
        st.success(f"μᵣ required ≈ {solved_value:.3f}")
else:
    st.info("Enter exactly three of the four fields to compute the missing quantity.")

st.divider()

st.header("Diameter vs. Skin Depth")
st.latex(r"\text{ratio} = \frac{d}{2\delta}")

col1, col2 = st.columns(2)
with col1:
    diameter = st.number_input("Conductor diameter d [mm]", value=1.0, format="%f")
with col2:
    delta_mm = st.number_input("Skin depth δ [mm]", value=(skin_val or 0) * 1e3, format="%f")

if delta_mm and delta_mm > 0:
    ratio = diameter / (2 * delta_mm)
    st.info(
        f"d/(2δ) ≈ {ratio:.2f} → "+
        ("DC-like (skin effect small)" if ratio < 0.5 else "Noticeable AC resistance; consider litz/parallel")
    )
else:
    st.caption("Compute δ above or enter it directly to see the ratio.")
