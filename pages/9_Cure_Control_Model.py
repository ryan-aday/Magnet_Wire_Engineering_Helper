import math
from pathlib import Path

import sympy as sp
import streamlit as st

st.title("Cure Control Model: Open vs. Closed Loop")

st.markdown(
    """
    Use these models to estimate enamel cure on a moving wire and to size line speed, oven
    temperature, or airflow targets. You can toggle between **open-loop** (report the cure
    index from entered conditions) and **closed-loop** (solve a setpoint to hit a target
    cure index) modes.
    """
)

st.subheader("Model walkthrough")
st.markdown(
    r"""
    ### 1. Simplified physics model of heating + cure

    **1.1 Variables**

    Let’s define the main knobs:

    * $L$ – heated oven length [m]
    * $v$ – line speed (draw speed) [m/s]
    * $T_{air}$ – oven air temperature [K] (or °C + 273.15)
    * $v_{air}$ – air velocity [m/s] → affects $h$
    * $d_w$ – overall wire diameter (metal + enamel) [m]
    * $t_e$ – enamel thickness (single side) [m]
    * $\rho_{{eff}},\; c_{{p,eff}}$ – effective density & specific heat of metal + coating
    * $h$ – convective heat-transfer coefficient [W/m²·K] (function of $v_{{air}}$, gas props)
    * $T_{in}$ – wire temperature entering oven [K]

    Cure kinetics (Arrhenius-type):

    $$k(T) = k_0 \exp\!\left(-\frac{E_a}{R T}\right) \quad [1/\text{s}]$$

    * $E_a$ – activation energy [J/mol]
    * $k_0$ – pre-exponential factor [1/s]
    * $R$ – gas constant [J/mol·K]

    We’ll define a dimensionless cure index with **design aim $C \approx 1$**:

    * $C = 1$ → “fully cured enough” (on-spec)
    * $C < 1$ → undercured (e.g., $C=0$ means no cure progress)
    * $C > 1$ → overbaked / excessively aged

    **1.2 Heating of a moving wire in the oven**

    Take a length element $dz$ of wire moving at speed $v$ through a zone with air at $T_{{air}}$.

    Energy balance (steady, 1D, lumped wire temp):

    $$\rho_{{eff}} A c_{{p,eff}} v \frac{dT_w}{dz} = h P (T_{{air}} - T_w)$$

    * $A = \tfrac{\pi d_w^2}{4}$ is cross-sectional area
    * $P = \pi d_w$ is perimeter

    Define

    $$K = \frac{hP}{\rho_{{eff}} A c_{{p,eff}} v}\;[1/\text{m}]$$

    Solution along the oven (with $z=0$ at entrance):

    $$T_w(z) = T_{{air}} - (T_{{air}} - T_{{in}}) e^{-K z}$$

    Thermal length scale:

    $$L_\theta = \frac{1}{K} = \frac{\rho_{{eff}} A c_{{p,eff}} v}{hP}$$

    If $L \gg L_\theta$, the wire essentially reaches air temperature.

    **1.3 Cure index along the oven**

    Residence time: $t_{{res}} = \tfrac{L}{v}$

    Degree of cure (index) from $0 \to L$:

    $$C = \int k(T_w(t))\,dt = \int \frac{k(T_w(z))}{v}\,dz$$

    Exact integral is ugly, so for systems engineering we usually use a temperature average:

    Average wire temperature along the oven:

    $$\bar{T}_w = T_{{air}} - (T_{{air}} - T_{{in}})\left(\frac{L_\theta}{L}\right)\left[1-e^{-L/L_\theta}\right]$$

    Use Arrhenius cure at this effective temperature:

    $$C \approx k(\bar{T}_w)\,t_{{res}} = k_0 \exp\!\left(-\frac{E_a}{R \bar{T}_w}\right) \frac{L}{v}$$

    This is your core systems equation.

    Note that $\bar{T}_w$ depends on $v, h, d_w$, etc. via $L_\theta$ and $h$ itself follows a Nu–Re–Pr correlation:

    $$\mathrm{Nu} = \frac{h d_w}{k_{{air}}} = C_1 \mathrm{Re}^m \mathrm{Pr}^n,\quad \mathrm{Re} = \frac{\rho_{{air}} v_{{air}} d_w}{\mu}$$

    So $C = F(T_{{air}}, v, v_{{air}}, d_w, t_e, \text{material props})$ and we enforce $C = C_{{target}} \approx 1$.

    ### 2. Solving for one parameter in terms of others

    You now have a single equation $C_{{target}} = k_0 \exp(-E_a/(R \bar{T}_w)) \tfrac{L}{v}$ with:

    * $\bar{T}_w = T_{{air}} - (T_{{air}} - T_{{in}})(L_\theta/L)[1-e^{-L/L_\theta}]$
    * $L_\theta = \tfrac{\rho_{{eff}} A c_{{p,eff}} v}{hP}$
    * $h = h(v_{{air}}, d_w, \text{air props})$

    Given numerical values for everything except one variable, solve for that unknown (analytically in simple cases, numerically in general):

    * **Line speed $v$ (common):** solve $0 = k_0 \exp(-E_a/(R \bar{T}_w(v))) \tfrac{L}{v} - C_{{target}}$ numerically. If $\bar{T}_w \approx T_{{air}}$, then $v \approx \tfrac{k_0 L}{C_{{target}}}\exp(-E_a/(R T_{{air}}))$.
    * **Oven temperature $T_{{air}}$:** with $\bar{T}_w \approx T_{{air}}$, $T_{{air}} = -\tfrac{E_a}{R \ln(C_{{target}} v/(k_0 L))}$ (log argument must be between 0 and 1).
    * **Air-side $h$ or $v_{{air}}$:** solve $0 = k_0 \exp(-E_a/(R \bar{T}_w(h))) \tfrac{L}{v} - C_{{target}}$ numerically (Nu–Re links $h$ and $v_{{air}}$).
    * **Enamel thickness / diameter:** $t_e$ shifts $\rho_{{eff}}, c_{{p,eff}}, A, P$; solve $C_{{target}} = F(T_{{air}}, v, v_{{air}}, d_w(t_e), t_e)$ for $t_e$.

    ### 3. Closed-loop control architecture

    **What you can actually measure**

    * Oven air temperatures (per zone)
    * Line speed
    * Air flow / pressure (to infer $h$)
    * Optional: wire surface temperature (IR pyrometer)
    * Optional: solvent concentration in exhaust (drying completeness)

    A cure estimator (these equations + parameters) produces $\hat{C}$, which feeds the outer loop.

    **Control loops (concept)**

    * Inner loops: PID on each $T_{{air,set}}$ and on airflow.
    * Outer loop: target $C_{{target}}$, compare to $\hat{C}$, and adjust line speed and/or setpoints.
    """
)

st.subheader("Governing Equations")
st.latex(r"\rho_{eff} A c_{p,eff} v \frac{dT_w}{dz} = h P (T_{air} - T_w)")
st.latex(r"K = \frac{h P}{\rho_{eff} A c_{p,eff} v}\;,\; L_\theta = \frac{1}{K} = \frac{\rho_{eff} A c_{p,eff} v}{h P}")
st.latex(r"T_w(z) = T_{air} - (T_{air} - T_{in}) e^{-K z}")
st.latex(
    r"\bar{T}_w = T_{air} - (T_{air} - T_{in})\left(\frac{L_\theta}{L}\right)\left[1 - e^{-L/L_\theta}\right]"
)
st.latex(r"C \approx k_0 e^{-E_a/(R \bar{T}_w)} \frac{L}{v}\quad (C \approx 1 \Rightarrow \text{on-spec cure})")

st.caption("Temperatures are in kelvin inside the kinetics; inputs below accept °C and convert internally.")

st.markdown(
    r"The controller sketch and solver follow the same flow described in the earlier LLM breakdown: "
    r"the moving-wire energy balance is solved to get the exponential temperature rise, the thermal "
    r"length scale \(L_\theta\) shapes the average temperature, and Arrhenius kinetics integrate "
    r"over residence time to form the cure index \(C\). Closed-loop control compares a target \(C\) "
    r"with the estimated \(\hat{C}\) from these equations and adjusts speed or setpoints."
)

st.subheader("How these equations form the model")
st.markdown(
    """
    The above narrative walks through the exact ODE, thermal length scale, averaged temperature,
    and Arrhenius kinetics that build the cure-index model. The calculators below directly use
    those relationships.
    """
)

st.info(
    r"""
    Gas constant \(R\) defaults to the universal 8.314 J/mol·K. The Arrhenius pair defaults to
    \(k_0 = 3.0\times10^4\,\text{s}^{-1}\) and \(E_a = 90\,\text{kJ/mol}\), tuned so the default
    inputs (380 °C air, 0.05 m/s line speed, 30 m heated length, ≈1 mm wire, \(h\) ≈ 500 W/m²·K)
    produce \(C \approx 1\). Edit all three to match your enamel data; the 380 °C default sits
    inside the ≈550 °F (288 °C) bake guidance noted by lpenamelwire while still demonstrating the model.
    """
)

st.markdown(
    r"""
    **How to identify \(k_0\) and \(E_a\) in practice**

    1. **Supplier kinetics or cure-time charts:** use “equivalent cure” points (e.g., 5 min @ 420 °C,
       10 min @ 400 °C) to fit an Arrhenius line: plot \(\ln t\) vs. \(1/T\); slope → \(E_a/R\),
       intercept → \(-\ln k_0 + \ln C_{target}\).
    2. **In-house cure or DSC tests:** measure time-to-cure at multiple temperatures (gel fraction,
       exotherm, modulus) and fit the same \(\ln t\) vs. \(1/T\) line to extract \(E_a\) and \(k_0\).
    3. **Control-oriented shortcut:** pick one proven oven condition \((T_{ref}, v_{ref})\) that yields
       acceptable cure, assume \(\bar{T}_w \approx T_{ref}\), and back-solve \(k_0\) for \(C \approx 1\)
       using your chosen \(E_a\); then refine both parameters against two or more additional test points.
    """
)

st.divider()

st.subheader("Wire & Oven Inputs")
col_geom1, col_geom2, col_geom3 = st.columns(3)
with col_geom1:
    d_bare_mm = st.number_input("Bare conductor diameter d_bare [mm]", value=1.0, min_value=0.01, step=0.01)
    t_enamel_um = st.number_input("Enamel thickness per side t_e [μm]", value=25.0, min_value=0.0, step=1.0)
    L_oven = st.number_input("Heated length L [m]", value=30.0, min_value=0.1)
with col_geom2:
    rho_eff = st.number_input("Effective density ρ_eff [kg/m³]", value=8900.0, min_value=100.0)
    cp_eff = st.number_input("Effective specific heat c_p,eff [J/kg-K]", value=385.0, min_value=10.0)
    T_in = st.number_input("Wire entry T_in [°C]", value=50.0)
with col_geom3:
    R_gas = st.number_input("Gas constant R [J/mol-K]", value=8.314, min_value=1.0, format="%f")
    k0 = st.number_input("Pre-exponential factor k0 [1/s]", value=30_000.0, min_value=0.0001, format="%f")
    Ea = st.number_input("Activation energy E_a [J/mol]", value=90000.0, min_value=1000.0)
    C_target = st.number_input("Target cure index C_target", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

# Derived dimensions
bare_m = d_bare_mm / 1000.0
d_overall = bare_m + 2 * (t_enamel_um / 1e6)
A_cross = math.pi * (d_overall ** 2) / 4
perimeter = math.pi * d_overall

mode = st.radio("Select mode", ["Open-loop (report cure index)", "Closed-loop (solve a setpoint)"])

# Defaults that downstream sections can re-use for the PID simulation
current_T_air = 380.0
current_v = 0.05
current_h = 500.0

st.subheader("Optional measurements")
use_pyro = st.checkbox("Use measured wire surface temperature (pyrometer)", value=False)
pyro_temp_c = (
    st.number_input("Measured wire surface temperature [°C]", value=180.0, min_value=-50.0)
    if use_pyro
    else None
)
use_solvent = st.checkbox("Use solvent concentration / evaporation completeness", value=False)
solvent_residual_pct = (
    st.number_input("Residual solvent indicator [% of baseline]", value=0.0, min_value=0.0, max_value=100.0)
    if use_solvent
    else 0.0
)
solvent_factor = max(0.0, 1.0 - solvent_residual_pct / 100.0)
st.caption("These optional measurements feed both the open- and closed-loop estimates.")


def average_wire_temp(T_air_c: float, v_mps: float, h_wm2k: float) -> tuple[float, float]:
    """Return average wire temperature in °C and thermal length L_theta [m]."""
    if h_wm2k <= 0 or v_mps <= 0:
        return math.nan, math.nan
    L_theta = (rho_eff * A_cross * cp_eff * v_mps) / (h_wm2k * perimeter)
    ratio = (L_theta / L_oven) * (1 - math.exp(-L_oven / L_theta))
    T_bar = T_air_c - (T_air_c - T_in) * ratio
    return T_bar, L_theta


def cure_index(T_air_c: float, v_mps: float, h_wm2k: float) -> tuple[float, float, float]:
    """Return cure index plus effective average temperature and L_theta."""
    T_bar_c_model, L_theta_val = average_wire_temp(T_air_c, v_mps, h_wm2k)
    if math.isnan(T_bar_c_model):
        return math.nan, math.nan, math.nan
    T_eff_c = pyro_temp_c if use_pyro else T_bar_c_model
    T_bar_k = T_eff_c + 273.15
    cure = float(k0 * math.exp(-Ea / (R_gas * T_bar_k)) * (L_oven / v_mps) * solvent_factor)
    return cure, T_eff_c, L_theta_val


class PIDController:
    """Minimal PID based on Digi-Key's Python tutorial."""

    def __init__(self, kp: float, ki: float, kd: float, out_min: float | None = None, out_max: float | None = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> tuple[float, float, float, float]:
        if dt <= 0:
            return 0.0, 0.0, self.integral, 0.0
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_min is not None:
            output = max(self.out_min, output)
        if self.out_max is not None:
            output = min(self.out_max, output)
        self.prev_error = error
        return output, error, self.integral, derivative


if "Open-loop" in mode:
    col_inputs1, col_inputs2, col_inputs3 = st.columns(3)
    with col_inputs1:
        T_air = st.number_input("Zone air temperature T_air [°C]", value=380.0)
    with col_inputs2:
        v_line = st.number_input("Line speed v [m/s]", value=0.05, min_value=0.00001, format="%f")
    with col_inputs3:
        h_coeff = st.number_input("Convection coefficient h [W/m²-K]", value=500.0, min_value=0.1)

    C_value, T_eff, L_theta_val = cure_index(T_air, v_line, h_coeff)
    current_T_air, current_v, current_h = T_air, v_line, h_coeff

    if not math.isnan(C_value):
        st.success(
            f"Effective wire temperature ≈ {T_eff:.1f} °C (L_theta ≈ {L_theta_val:.2f} m); "
            f"cure index C ≈ {C_value:.3f} (target {C_target})"
            + (" using pyrometer T" if use_pyro else "")
        )
    else:
        st.warning("Enter positive speed and convection to compute the cure index.")

else:
    st.markdown("Choose which setpoint to solve for to achieve the target cure index.")
    solve_choice = st.selectbox("Solve for", ["Line speed v", "Air temperature T_air", "Convection coefficient h"])

    # Shared knowns
    T_air_guess = st.number_input("Air temperature guess T_air [°C]", value=380.0)
    v_guess = st.number_input("Line speed guess v [m/s]", value=0.05, min_value=0.00001, format="%f")
    h_guess = st.number_input("Convection guess h [W/m²-K]", value=500.0, min_value=0.1)
    current_T_air, current_v, current_h = T_air_guess, v_guess, h_guess

    def solve_for_variable(symbol: sp.Symbol, expr: sp.Expr, guess: float) -> float | None:
        """Try multiple seeds plus coarse log-spaced seeds to avoid nsolve dead-ends."""

        def unique_positive(seq: list[float]) -> list[float]:
            seen = set()
            ordered = []
            for val in seq:
                if val is None:
                    continue
                if val <= 0:
                    continue
                key = round(val, 12)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(val)
            return ordered

        seeds = [guess, guess * 0.5, guess * 2, guess * 0.1, guess * 5, guess + 1.0, max(0.1, guess - 1.0)]
        seeds.extend([10 ** (p / 2) for p in range(-8, 11)])
        seeds = unique_positive(seeds)

        for g in seeds:
            try:
                sol = sp.nsolve(expr - C_target, g, tol=1e-9, maxsteps=300, prec=80)
                sol_val = float(sol)
                if sol_val > 0:
                    return sol_val
            except Exception:
                continue
        return None

    if solve_choice == "Line speed v":
        v_sym = sp.symbols("v", positive=True)
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_sym) / (h_guess * perimeter)
        if use_pyro:
            Tbar_expr = pyro_temp_c
        else:
            Tbar_expr = T_air_guess - (T_air_guess - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_gas * (Tbar_expr + 273.15))) * (L_oven / v_sym)
        solved = solve_for_variable(v_sym, C_expr, v_guess)
        if solved:
            st.success(f"Required line speed v ≈ {solved:.4f} m/s to hit C_target")
        else:
            st.error("Could not solve for line speed with the provided guesses.")
    elif solve_choice == "Air temperature T_air":
        T_sym = sp.symbols("Tair")
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_guess * perimeter)
        if use_pyro:
            st.info("Disable the pyrometer override to solve for T_air; the measured temperature fixes \bar{T}_w.")
            Tbar_expr = pyro_temp_c
        else:
            Tbar_expr = T_sym - (T_sym - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_gas * (Tbar_expr + 273.15))) * (L_oven / v_guess)
        solved = solve_for_variable(T_sym, C_expr, T_air_guess)
        if solved:
            st.success(f"Required air temperature T_air ≈ {solved:.1f} °C to hit C_target")
        else:
            st.error("Could not solve for air temperature with the provided guesses.")
    else:
        h_sym = sp.symbols("h", positive=True)
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_sym * perimeter)
        if use_pyro:
            st.info("Disable the pyrometer override to solve for h; the measured temperature fixes \bar{T}_w.")
            Tbar_expr = pyro_temp_c
        else:
            Tbar_expr = T_air_guess - (T_air_guess - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_gas * (Tbar_expr + 273.15))) * (L_oven / v_guess)
        solved = solve_for_variable(h_sym, C_expr, h_guess)
        if solved:
            st.success(f"Required convection coefficient h ≈ {solved:.1f} W/m²-K to hit C_target")
        else:
            st.error("Could not solve for convection coefficient with the provided guesses.")

st.divider()

st.subheader("PID helper (from Digi-Key tutorial)")
st.markdown(
    "Use this lightweight PID stepper to prototype an outer-loop controller that trims line speed or"
    " oven setpoints. Algorithm structure follows the Digi-Key guide: "
    "https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python"
)

if "pid_state" not in st.session_state:
    st.session_state.pid_state = {"integral": 0.0, "prev_error": 0.0}

pid_col1, pid_col2, pid_col3 = st.columns(3)
with pid_col1:
    pid_setpoint = st.number_input("Setpoint (e.g., C_target or speed)", value=1.0, format="%f")
    pid_meas = st.number_input("Measured value", value=0.9, format="%f")
    pid_dt = st.number_input("Loop interval dt [s]", value=0.5, min_value=0.0001, format="%f")
with pid_col2:
    kp = st.number_input("Kp", value=1.0, format="%f")
    ki = st.number_input("Ki", value=0.1, format="%f")
    kd = st.number_input("Kd", value=0.01, format="%f")
with pid_col3:
    use_min = st.checkbox("Clamp min?", value=False)
    out_min = st.number_input("Output min", value=0.0) if use_min else None
    use_max = st.checkbox("Clamp max?", value=False)
    out_max = st.number_input("Output max", value=2.0) if use_max else None
    if st.button("Reset PID state"):
        st.session_state.pid_state = {"integral": 0.0, "prev_error": 0.0}

controller = PIDController(
    kp,
    ki,
    kd,
    out_min=out_min,
    out_max=out_max,
)
controller.integral = st.session_state.pid_state["integral"]
controller.prev_error = st.session_state.pid_state["prev_error"]

pid_output, err, integ, deriv = controller.update(pid_setpoint, pid_meas, pid_dt)
st.session_state.pid_state = {"integral": integ, "prev_error": err}

st.info(
    f"PID output: {pid_output:.4f} (error={err:.4f}, integral={integ:.4f}, derivative={deriv:.4f}). "
    "Apply this as a speed or setpoint trim in the closed-loop concept."
)

st.subheader("PID simulation toward C ≈ 1")
st.markdown(
    """
    Prototype how the outer-loop PID drives the cure index toward the target using a simple
    discrete simulation. The PID output nudges one chosen actuator (speed, temperature, or
    convection) each step while the cure index is recomputed from the moving-wire model.
    """
)

sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
with sim_col1:
    sim_steps = st.number_input("Simulation steps", value=60, min_value=5, step=5)
    sim_dt = st.number_input("Simulation dt [s]", value=0.5, min_value=0.01, format="%f")
with sim_col2:
    control_choice = st.selectbox("Control variable", ["Line speed v", "Air temperature T_air", "Convection h"])
    control_gain = st.number_input("Actuator gain (units per PID unit)", value=0.1, format="%f")
with sim_col3:
    v_min = st.number_input("Min line speed [m/s]", value=0.05, min_value=0.00001, format="%f")
    h_min = st.number_input("Min h [W/m²-K]", value=1.0, min_value=0.001, format="%f")
with sim_col4:
    T_min = st.number_input("Min T_air [°C]", value=25.0)
    T_max = st.number_input("Max T_air [°C]", value=600.0)

sim_pid = PIDController(kp, ki, kd, out_min=out_min, out_max=out_max)
sim_pid.reset()

sim_records = []
v_sim, T_sim, h_sim = current_v, current_T_air, current_h

for step in range(int(sim_steps)):
    C_meas, _, _ = cure_index(T_sim, v_sim, h_sim)
    if math.isnan(C_meas):
        st.error("Simulation halted: invalid cure index (check inputs and mins).")
        break
    pid_out, err_sim, integ_sim, deriv_sim = sim_pid.update(C_target, C_meas, sim_dt)

    if control_choice == "Line speed v":
        v_sim = max(v_min, v_sim + control_gain * pid_out * sim_dt)
    elif control_choice == "Air temperature T_air":
        T_sim = min(T_max, max(T_min, T_sim + control_gain * pid_out * sim_dt))
    else:
        h_sim = max(h_min, h_sim + control_gain * pid_out * sim_dt)

    sim_records.append(
        {
            "time_s": step * sim_dt,
            "Cure index": C_meas,
            "Control": v_sim if control_choice == "Line speed v" else (T_sim if control_choice == "Air temperature T_air" else h_sim),
        }
    )

if sim_records:
    import pandas as pd

    sim_df = pd.DataFrame(sim_records)
    st.line_chart(sim_df.set_index("time_s"), use_container_width=True)
    st.caption(
        "Control trace shows the manipulated variable evolution; cure index is driven toward the target (default 1.0)."
    )

st.subheader("Closed-loop block diagram")
st.caption("Illustrative flow mirroring the provided cure-control schematic.")
diagram_path = Path("data/system_diagram.png")
if diagram_path.exists():
    st.image(str(diagram_path), caption="Closed-loop cure control schematic", use_container_width=True)
else:
    st.warning("system_diagram.png not found in data/. Add the provided diagram to display it here.")
