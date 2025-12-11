import math
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

st.title("Wire Basics & DC Resistance")

st.markdown(
    """
    Quick geometry and DC calculations for magnet wire. Approximations follow common AWG
    relationships and simple resistivity models. Provide all but one variable in each calculator
    to solve for the missing value.
    """
)

# Helper functions

def solve_missing(values):
    missing_keys = [k for k, v in values.items() if v is None]
    if len(missing_keys) != 1:
        return None, None
    missing_key = missing_keys[0]
    return missing_key, {k: v for k, v in values.items() if v is not None}


st.header("AWG, Diameter, and Area")
st.latex(r"d_{mm} \approx 0.127 \times 92^{\frac{36-\text{AWG}}{39}}")
st.latex(r"A = \frac{\pi d^2}{4}")

col1, col2, col3 = st.columns(3)
with col1:
    awg_in = st.number_input("AWG (leave blank to solve)", min_value=0.0, max_value=50.0, step=1.0, format="%f")
    awg_value = awg_in if awg_in > 0 else None
with col2:
    diameter_mm = st.number_input("Bare diameter d [mm] (blank to solve)", min_value=0.0, step=0.01, format="%f")
    diameter_value = diameter_mm if diameter_mm > 0 else None
with col3:
    area_mm2 = st.number_input("Area A [mm²] (blank to solve)", min_value=0.0, step=0.01, format="%f")
    area_value = area_mm2 if area_mm2 > 0 else None

# Determine which variable to solve
provided = {"awg": awg_value, "diameter": diameter_value, "area": area_value}
missing_key, known = solve_missing(provided)

if missing_key == "diameter":
    diameter = 0.127 * 92 ** ((36 - known["awg"]) / 39)
    st.success(f"Diameter ≈ {diameter:.4f} mm")
    area = math.pi * (diameter / 2) ** 2
    st.info(f"Area ≈ {area:.4f} mm²")
elif missing_key == "awg":
    awg = 36 - 39 * math.log10(known["diameter"] / 0.127) / math.log10(92)
    st.success(f"AWG ≈ {awg:.1f}")
    area = math.pi * (known["diameter"] / 2) ** 2
    st.info(f"Area ≈ {area:.4f} mm²")
elif missing_key == "area":
    area = math.pi * (known["diameter"] / 2) ** 2
    st.success(f"Area ≈ {area:.4f} mm²")
else:
    if all(v is not None for v in provided.values()):
        st.info("All variables provided; nothing to solve. Adjust a field to see calculations.")


st.header("DC Resistance (20°C baseline)")
st.latex(r"R = \frac{\rho L}{A}")

rho = st.number_input("Resistivity ρ [Ω·m]", value=1.724e-8, format="%e")
length = st.number_input("Length L [m]", value=10.0)
area_res = st.number_input("Area A [m²] (bare area)", value=1e-6, format="%e")
resistance = st.number_input("Resistance R [Ω] (leave 0 to solve)", value=0.0, format="%f")

R, rho_sym, L, A = sp.symbols("R rho L A", positive=True)
equation = sp.Eq(R, rho_sym * L / A)
symbol_values = {R: resistance if resistance > 0 else None, rho_sym: rho, L: length, A: area_res}
missing_symbol, solved_value = solve_missing_symbol(equation, symbol_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == R:
        st.success(f"R ≈ {solved_value:.6f} Ω")
    elif missing_symbol == A:
        st.success(f"Area required ≈ {solved_value:.6e} m²")
    elif missing_symbol == L:
        st.success(f"Length allowed ≈ {solved_value:.3f} m")
else:
    st.info("Provide exactly one blank to solve for the missing variable.")


st.header("Resistance vs. Temperature")
st.latex(r"R(T) = R_{20}[1 + \alpha (T - 20^\circ C)]")

col1, col2, col3 = st.columns(3)
with col1:
    r20 = st.number_input("R₍₂₀₎ [Ω] (blank to solve)", value=0.0, format="%f")
    r20_val = r20 if r20 > 0 else None
with col2:
    alpha = st.number_input("Temperature coefficient α [1/K]", value=0.00393, format="%f")
with col3:
    temp_c = st.number_input("Temperature T [°C]", value=80.0)

rt = st.number_input("R(T) [Ω] (blank to solve)", value=0.0, format="%f")
rt_val = rt if rt > 0 else None

R20_sym, alpha_sym, T_sym, RT_sym = sp.symbols("R20 alpha T RT")
temp_equation = sp.Eq(RT_sym, R20_sym * (1 + alpha_sym * (T_sym - 20)))
temp_values = {R20_sym: r20_val, alpha_sym: alpha, T_sym: temp_c, RT_sym: rt_val}
missing_symbol, solved_value = solve_missing_symbol(temp_equation, temp_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == RT_sym:
        st.success(f"R(T) ≈ {solved_value:.6f} Ω")
    elif missing_symbol == R20_sym:
        st.success(f"R₍₂₀₎ ≈ {solved_value:.6f} Ω")
    elif missing_symbol == T_sym:
        st.success(f"Temperature ≈ {solved_value:.2f} °C")
else:
    st.info("Enter three of the four values to solve the remaining one.")


st.header("Current Density")
st.latex(r"J = \frac{I}{A}")

col1, col2 = st.columns(2)
with col1:
    current = st.number_input("Current I [A] (blank to solve)", value=0.0, format="%f")
    current_val = current if current > 0 else None
with col2:
    area_j = st.number_input("Area A [mm²] (blank to solve)", value=0.0, format="%f")
    area_j_val = area_j if area_j > 0 else None

j_input = st.number_input("Current density J [A/mm²] (blank to solve)", value=0.0, format="%f")
j_val = j_input if j_input > 0 else None

I_sym, A_sym, J_sym = sp.symbols("I A J", positive=True)
j_equation = sp.Eq(J_sym, I_sym / A_sym)
j_values = {I_sym: current_val, A_sym: area_j_val, J_sym: j_val}
missing_symbol, solved_value = solve_missing_symbol(j_equation, j_values)

if missing_symbol is not None and solved_value is not None:
    if missing_symbol == J_sym:
        st.success(f"J ≈ {solved_value:.3f} A/mm²")
    elif missing_symbol == I_sym:
        st.success(f"Current ≈ {solved_value:.3f} A")
    elif missing_symbol == A_sym:
        st.success(f"Area required ≈ {solved_value:.3f} mm²")
else:
    st.info("Provide two fields and leave one blank to compute the missing quantity.")
