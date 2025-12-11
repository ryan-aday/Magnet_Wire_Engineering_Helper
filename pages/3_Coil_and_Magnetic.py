import math
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

MU0 = 4 * math.pi * 1e-7

st.title("Coil, Inductance, and Magnetic Checks")

st.caption(
    "Permeability constant μ₀ = 4π × 10⁻⁷ H/m (Engineering Toolbox: "
    "https://www.engineeringtoolbox.com/permeability-d_1923.html)."
)

st.markdown(
    """
    Calculate inductance for simple cores, flux density, and basic electromagnetic force. Supply
    n−1 variables to solve for the remaining one wherever possible.
    """
)
st.header("Long Solenoid / Core Inductance")
st.latex(r"L = \mu_0 \mu_r \frac{N^2 A}{\ell_m}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    turns = st.number_input("Turns N (blank to solve)", value=0.0, format="%f")
    turns_val = turns if turns > 0 else None
with col2:
    area = st.number_input("Core cross-section A [m²]", value=1e-4, format="%e")
with col3:
    path_length = st.number_input("Magnetic path length ℓₘ [m]", value=0.1, format="%f")
with col4:
    mu_r = st.number_input("Relative permeability μᵣ", value=1000.0, format="%f")

inductance = st.number_input("Inductance L [H] (blank to solve)", value=0.0, format="%e")
inductance_val = inductance if inductance > 0 else None

L_sym, mu_r_sym, N_sym, A_sym, lm_sym = sp.symbols("L mu_r N A lm", positive=True)
equation = sp.Eq(L_sym, MU0 * mu_r_sym * N_sym**2 * A_sym / lm_sym)
symbol_values = {L_sym: inductance_val, mu_r_sym: mu_r, N_sym: turns_val, A_sym: area, lm_sym: path_length}
missing_symbol, solved_value = solve_missing_symbol(equation, symbol_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == L_sym:
        st.success(f"L ≈ {solved_value:.6e} H")
    elif missing_symbol == N_sym:
        st.success(f"Turns N ≈ {solved_value:.2f}")
    elif missing_symbol == A_sym:
        st.success(f"Area A ≈ {solved_value:.3e} m²")
    elif missing_symbol == lm_sym:
        st.success(f"ℓₘ ≈ {solved_value:.3f} m")
else:
    st.info("Provide four of the five fields to solve for the missing one.")

st.divider()

st.header("Toroid Inductance")
st.latex(r"L = \mu_0 \mu_r \frac{N^2 A}{2\pi r_{\text{mean}}}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    t_turns = st.number_input("Turns N (toroid)", value=20.0)
with col2:
    t_area = st.number_input("Core cross-section A [m²] (toroid)", value=1e-4, format="%e")
with col3:
    r_mean = st.number_input("Mean radius r_mean [m]", value=0.03)
with col4:
    t_mu_r = st.number_input("Relative permeability μᵣ (toroid)", value=2000.0)

toroid_L = MU0 * t_mu_r * (t_turns ** 2) * t_area / (2 * math.pi * r_mean)
st.success(f"Toroid L ≈ {toroid_L:.6e} H")

st.divider()

st.header("Flux Density and Flux")
st.latex(r"B = \mu_0 \mu_r \frac{N I}{\ell_m}, \quad \Phi = B A")

col1, col2, col3 = st.columns(3)
with col1:
    current = st.number_input("Current I [A]", value=1.0)
with col2:
    turns_flux = st.number_input("Turns N", value=10.0)
with col3:
    lm_flux = st.number_input("Path length ℓₘ [m]", value=0.05)

mu_r_flux = st.number_input("Relative permeability μᵣ (flux)", value=1500.0)
area_flux = st.number_input("Area A [m²] (flux)", value=5e-4, format="%e")

B = MU0 * mu_r_flux * turns_flux * current / lm_flux
phi = B * area_flux

st.success(f"Flux density B ≈ {B:.3f} T")
st.info(f"Flux Φ ≈ {phi:.6e} Wb")

st.caption("Compare B against material saturation to ensure margin.")

st.divider()

st.header("Force on a Straight Conductor")
st.latex(r"F = B I \ell")

col1, col2, col3 = st.columns(3)
with col1:
    B_force = st.number_input("Field B [T]", value=B)
with col2:
    I_force = st.number_input("Current I [A] (force)", value=current)
with col3:
    length_force = st.number_input("Conductor length ℓ [m]", value=0.1)

force = B_force * I_force * length_force
st.success(f"Force ≈ {force:.3f} N")
