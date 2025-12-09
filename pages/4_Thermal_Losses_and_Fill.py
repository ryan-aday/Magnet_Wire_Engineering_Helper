import math
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

st.title("Thermal, Losses, and Slot Fill")

st.markdown(
    """
    Estimate copper loss, temperature rise, insulation life factor, and slot/winding fill. Supply
    n−1 variables to solve the unknown in each calculator.
    """
)

st.header("Copper Loss")
st.latex(r"P_{Cu} = I_{rms}^2 R_{AC}")

col1, col2, col3 = st.columns(3)
with col1:
    i_rms = st.number_input("I_rms [A] (blank to solve)", value=0.0, format="%f")
    i_val = i_rms if i_rms > 0 else None
with col2:
    r_ac = st.number_input("R_AC [Ω] (blank to solve)", value=0.0, format="%f")
    r_val = r_ac if r_ac > 0 else None
with col3:
    p_cu = st.number_input("P_Cu [W] (blank to solve)", value=0.0, format="%f")
    p_val = p_cu if p_cu > 0 else None

I_sym, R_sym, P_sym = sp.symbols("I R P", positive=True)
equation = sp.Eq(P_sym, I_sym**2 * R_sym)
symbol_values = {I_sym: i_val, R_sym: r_val, P_sym: p_val}
missing_symbol, solved_value = solve_missing_symbol(equation, symbol_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == P_sym:
        st.success(f"P_Cu ≈ {solved_value:.3f} W")
    elif missing_symbol == I_sym:
        st.success(f"I_rms ≈ {solved_value:.3f} A")
    elif missing_symbol == R_sym:
        st.success(f"R_AC ≈ {solved_value:.4f} Ω")
else:
    st.info("Provide two of three values to compute the missing one.")

st.divider()

st.header("Temperature Rise")
st.latex(r"\Delta T = P_{loss} R_{\theta}\quad;\quad T_{hot} = T_{amb} + \Delta T")

col1, col2, col3 = st.columns(3)
with col1:
    p_loss = st.number_input("Total losses P_loss [W]", value=p_val or 10.0)
with col2:
    r_theta = st.number_input("Thermal resistance Rθ [K/W] (blank to solve)", value=0.0, format="%f")
    r_theta_val = r_theta if r_theta > 0 else None
with col3:
    delta_t_input = st.number_input("ΔT [K] (blank to solve)", value=0.0, format="%f")
    delta_t_val = delta_t_input if delta_t_input > 0 else None

ambient = st.number_input("Ambient T [°C]", value=25.0)

P_loss_sym, Rtheta_sym, DeltaT_sym = sp.symbols("P_loss Rtheta DeltaT", positive=True)
equation_temp = sp.Eq(DeltaT_sym, P_loss_sym * Rtheta_sym)
temp_values = {P_loss_sym: p_loss, Rtheta_sym: r_theta_val, DeltaT_sym: delta_t_val}
missing_symbol, solved_value = solve_missing_symbol(equation_temp, temp_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == DeltaT_sym:
        st.success(f"ΔT ≈ {solved_value:.2f} K; T_hot ≈ {ambient + solved_value:.1f} °C")
    elif missing_symbol == Rtheta_sym:
        st.success(f"Rθ ≈ {solved_value:.3f} K/W")
else:
    st.info("Provide two of the three fields (P_loss, Rθ, ΔT) to solve the other and T_hot.")

st.divider()

st.header("Insulation Life (Arrhenius-style)")
st.latex(r"L = L_0 \exp\left[\frac{E_a}{k}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right]")

col1, col2, col3 = st.columns(3)
with col1:
    L0 = st.number_input("Baseline life L₀ [hours]", value=20000.0)
with col2:
    T0 = st.number_input("Baseline temp T₀ [K]", value=433.0, help="Example: 160°C class → 433 K")
with col3:
    Ea_over_k = st.number_input("Ea/k [K] (activation/boltzmann)", value=7000.0, help="Use 7000–9000 K as a rule of thumb")

operating_temp_c = st.number_input("Operating temp [°C]", value=150.0)
T_use = operating_temp_c + 273.15
life = L0 * math.exp(Ea_over_k * (1 / T_use - 1 / T0))
st.success(f"Estimated life ≈ {life:,.0f} hours (rule-of-thumb Arrhenius)")

st.caption("Lowering hotspot temperature by ~10 K roughly doubles life for many enamel systems.")

st.divider()

st.header("Slot / Winding Fill Factor")
st.latex(r"k_{fill} = \frac{A_{Cu, total}}{A_{slot}}")

col1, col2, col3 = st.columns(3)
with col1:
    slot_area = st.number_input("Slot area A_slot [mm²]", value=200.0)
with col2:
    conductor_area = st.number_input("Bare conductor area [mm²]", value=2.5)
with col3:
    conductor_count = st.number_input("Number of conductors", value=40, step=1)

total_cu = conductor_area * conductor_count
fill = total_cu / slot_area if slot_area > 0 else 0
st.success(f"Fill factor ≈ {fill:.2f}")

if fill < 0.3:
    st.info("Plenty of space; mechanical support and vibrations may dominate.")
elif fill < 0.5:
    st.info("Typical practical range for round wire windings.")
else:
    st.warning("High fill factor—check manufacturability, insulation clearances, and cooling.")
