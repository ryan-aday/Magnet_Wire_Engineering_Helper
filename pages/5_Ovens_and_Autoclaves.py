import math
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

st.title("Enameling Ovens & Autoclaves")

st.markdown(
    """
    Context for thermal equipment used with magnet wire:
    * **Continuous enameling ovens** build insulation layer-by-layer on bare wire (flash off solvent,
      cure polymer, cool). Adjustable by line speed, zone temperature, and air velocity.
    * **Annealing furnaces** soften Cu/Al ahead of coating.
    * **Autoclaves/VPI** are downstream in motor/transformer shops for vacuum-pressure impregnation
      of wound coils, followed by oven cure.
    """
)

st.header("Residence Time per Pass")
st.latex(r"t_{res} = \frac{L_{oven}}{v}")

col1, col2, col3 = st.columns(3)
with col1:
    L = st.number_input("Heated length L_oven [m]", value=30.0)
with col2:
    v = st.number_input("Line speed v [m/s] (blank to solve)", value=0.0, format="%f")
    v_val = v if v > 0 else None
with col3:
    t_res_in = st.number_input("Residence time t_res [s] (blank to solve)", value=0.0, format="%f")
    t_res_val = t_res_in if t_res_in > 0 else None

L_sym, v_sym, t_sym = sp.symbols("L v t", positive=True)
equation_res = sp.Eq(t_sym, L_sym / v_sym)
res_values = {L_sym: L, v_sym: v_val, t_sym: t_res_val}
missing_symbol, solved_value = solve_missing_symbol(equation_res, res_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == t_sym:
        st.success(f"t_res ≈ {solved_value:.2f} s")
    elif missing_symbol == v_sym:
        st.success(f"Line speed v ≈ {solved_value:.3f} m/s")
else:
    st.info("Leave either speed or time blank to solve for it.")

st.divider()

st.header("Lumped Heating per Oven Zone")
st.latex(r"T_w(t) = T_{air} - (T_{air} - T_{in}) e^{-t/\tau}\quad;\quad \tau = \frac{m c_p}{h A}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    T_air = st.number_input("Zone air temp T_air [°C]", value=420.0)
with col2:
    T_in = st.number_input("Wire entry temp T_in [°C]", value=50.0)
with col3:
    tau = st.number_input("Time constant τ [s] (blank to solve)", value=0.0, format="%f")
    tau_val = tau if tau > 0 else None
with col4:
    time_in_zone = st.number_input("Time in zone t [s] (blank to solve)", value=t_res_val or 5.0, format="%f")
    time_val = time_in_zone if time_in_zone > 0 else None

T_out_in = st.number_input("Wire exit temp T_w(t) [°C] (blank to solve)", value=0.0, format="%f")
T_out_val = T_out_in if T_out_in > 0 else None

Tair_sym, Tin_sym, tau_sym, t_zone_sym, Tout_sym = sp.symbols("Tair Tin tau t_zone Tout")
heating_equation = sp.Eq(Tout_sym, Tair_sym - (Tair_sym - Tin_sym) * sp.exp(-t_zone_sym / tau_sym))
heating_values = {
    Tair_sym: T_air,
    Tin_sym: T_in,
    tau_sym: tau_val,
    t_zone_sym: time_val,
    Tout_sym: T_out_val,
}
missing_symbol, solved_value = solve_missing_symbol(heating_equation, heating_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == Tout_sym:
        st.success(f"Wire exit temperature ≈ {solved_value:.1f} °C")
    elif missing_symbol == tau_sym:
        st.success(f"τ ≈ {solved_value:.2f} s")
    elif missing_symbol == t_zone_sym:
        st.success(f"Time in zone needed ≈ {solved_value:.2f} s")
else:
    st.info("Enter four of the five fields to solve the remaining one.")

st.caption(
    "Use τ = m c_p /(hA) if you know wire mass, specific heat, convection coefficient, and surface area."
)

st.divider()

st.header("Cure Time vs. Temperature (Arrhenius Equivalent)")
st.latex(r"t_2 = t_1 \exp\left[\frac{E_a}{R}\left(\frac{1}{T_2} - \frac{1}{T_1}\right)\right]")

col1, col2, col3 = st.columns(3)
with col1:
    t1 = st.number_input("Reference cure time t1 [s]", value=300.0)
with col2:
    T1 = st.number_input("Reference temp T1 [K]", value=673.0)
with col3:
    Ea = st.number_input("Activation energy Ea [J/mol]", value=90000.0)

T2 = st.number_input("New temp T2 [K]", value=693.0)
R = 8.314

t1_sym, T1_sym, Ea_sym, T2_sym, t2_sym = sp.symbols("t1 T1 Ea T2 t2", positive=True)
cure_equation = sp.Eq(t2_sym, t1_sym * sp.exp(Ea_sym / R * (1 / T2_sym - 1 / T1_sym)))
cure_values = {t1_sym: t1, T1_sym: T1, Ea_sym: Ea, T2_sym: T2, t2_sym: None}
_, t2_value = solve_missing_symbol(cure_equation, cure_values)

if t2_value is not None:
    st.success(f"Equivalent cure time t2 ≈ {t2_value:.1f} s at T2")

st.divider()

st.header("Autoclave / VPI Cycle Reminders")
st.markdown(
    """
    * Typical steps: **vacuum pull** → **resin fill** → **over-pressure hold (2–6 bar)** → **drain** →
      **oven cure**.
    * Track: vacuum level & duration, fill time, pressure, cure temperature/time.
    * VPI is for wound coils, whereas enameling ovens handle continuous bare wire coating.
    """
)
