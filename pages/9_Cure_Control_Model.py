import math
from pathlib import Path

import sympy as sp
import streamlit as st

R_GAS = 8.314  # J/mol-K

st.title("Cure Control Model: Open vs. Closed Loop")

st.markdown(
    """
    Use these models to estimate enamel cure on a moving wire and to size line speed, oven
    temperature, or airflow targets. You can toggle between **open-loop** (report the cure
    index from entered conditions) and **closed-loop** (solve a setpoint to hit a target
    cure index) modes.
    """
)

st.subheader("LLM-derived model walkthrough")
st.markdown(
    """
    1. Simplified physics model of heating + cure
    1.1 Variables

    Let’s define the main knobs:

    L – heated oven length [m]

    v – line speed (draw speed) [m/s]

    Tair – oven air temperature [K] (or °C + 273.15)

    vair – air velocity [m/s] → affects h

    dw – overall wire diameter (metal + enamel) [m]

    te – enamel thickness (single side) [m]

    ρeff, cp,eff – effective density & cp of metal + coating

    h – convective heat-transfer coefficient [W/m²·K] (function of vair, gas props)

    Tin – wire temperature entering oven [K]

    Cure kinetics (Arrhenius-type):

    k(T)=k0exp(−Ea/(R T)) [1/s]

    Ea – activation energy [J/mol]

    k0 – pre-exponential factor [1/s]

    R – gas constant [J/mol·K]

    We’ll define a dimensionless cure index:

    C=1 → “fully cured enough” (your spec)

    C<1 → undercured, C>1 → overbaked / excessively aged.

    1.2 Heating of a moving wire in the oven

    Take a length element dz of wire moving at speed v through a zone with air at Tair.

    Energy balance (steady, 1D, lumped wire temp):

    ρeff A cp,eff v dTw/dz = h P (Tair − Tw)

    A = π dw² /4 is cross-sectional area

    P = π dw is perimeter

    Define

    K = hP/(ρeff A cp,eff v) [1/m]

    Solution along the oven (with z=0 at entrance):

    Tw(z) = Tair − (Tair − Tin) e^(−K z)

    Thermal length scale:

    Lθ = 1/K = ρeff A cp,eff v/(hP)

    If L ≫ Lθ, the wire essentially reaches air temperature.

    1.3 Cure index along the oven

    Residence time:

    tres = L/v

    Degree of cure (index) from 0→L:

    C = ∫ k(Tw(t)) dt = ∫ k(Tw(z))/v dz

    Exact integral is ugly, so for systems engineering we usually use a temperature average:

    Compute average wire temperature along the oven:

    T̄w = (1/L)∫Tw(z) dz = Tair − (Tair − Tin)(Lθ/L)[1−e^(−L/Lθ)]

    Use Arrhenius cure at this effective temperature:

    C ≈ k(T̄w) tres = k0 exp(−Ea/(R T̄w)) L/v

    This is your core systems equation.

    Now note:

    T̄w depends on v, h, dw, etc. via Lθ.

    h is itself a function of air flow / air velocity and gas properties:

    Nu = h dw/kair = C1 Re^m Pr^n, Re = ρair vair dw / μ

    So you can view C as:

    C = F(Tair, v, vair, dw, te, material props)

    And enforce:

    C = Ctarget (≈1)

    This is the systems engineering constraint your controller must satisfy.

    2. Solving for one parameter in terms of others

    You now have a single equation:

    Ctarget = k0 exp(−Ea/(R T̄w)) L/v

    with

    T̄w = Tair − (Tair − Tin)(Lθ/L)[1−e^(−L/Lθ)]

    Lθ = ρeff A cp,eff v/(hP)

    h = h(vair, dw, air props)

    Given numerical values for everything except ONE variable, you can solve for that unknown (analytically in simple cases, numerically in general). A few examples:

    2.1 Solve for line speed v (common design case)

    In full form, v appears inside both T̄w and the L/v term, so you solve numerically:

    0 = k0 exp(−Ea/(R T̄w(v))) L/v − Ctarget

    Use a root-finder (Newton, bisection) in your Python control script.

    For quick design approximations (long oven, small wire) you can assume T̄w ≈ Tair. Then:

    Ctarget ≈ k0 exp(−Ea/(R Tair)) L/v

    So:

    v ≈ k0 L/Ctarget exp(−Ea/(R Tair))

    All the geometry & air-side stuff is then lumped into k0 (you can treat an “effective” k0 as a calibrated parameter).

    2.2 Solve for oven temperature Tair

    Using the simplified model (T̄w ≈ Tair):

    Ctarget = k0 exp(−Ea/(R Tair)) L/v

    Rearrange for Tair:

    exp(−Ea/(R Tair)) = Ctarget v/(k0 L)

    Tair = −Ea/(R ln(Ctarget v/(k0 L)))

    Valid as long as the log argument is between 0 and 1 (physically, you’re in the feasible region).

    2.3 Solve for air-side coefficient h or air velocity vair

    From the full average temperature expression, h appears in Lθ and thus in T̄w. So for target cure:

    0 = k0 exp(−Ea/(R T̄w(h))) L/v − Ctarget

    Again, numerically solve for h (or vair, via the Nu–Re correlation) given everything else.

    2.4 Enamel thickness / wire diameter effects

    Enamel thickness changes:

    • The thermal mass (via ρeff, cp,eff, A)
    • The surface exposure (via dw → perimeter P)
    • Potentially cure kinetics (if you distinguish bulk vs near-surface cure)

    In the model, these all enter through Lθ, hence T̄w, hence the cure index.

    So, for example, if you want to find the max enamel thickness you can cure at given T, v, and air flow, you again numerically solve:

    Ctarget = F(Tair, v, vair, dw(te), te)

    for te.

    3. Closed-loop control architecture
    3.1 What you can actually measure

    Real-world “cure” is hard to measure directly in real time, so you typically estimate it from:

    • Measured oven air temps in each zone
    • Measured line speed
    • Measured air flow / pressure (to infer h)
    • Optional: wire surface temperature via IR pyrometer
    • Optional: solvent concentration in exhaust (indicates drying completeness)

    From these, a Cure Estimator (your model + parameters) calculates a cure index estimate Ĉ.

    That becomes the process variable in the outer loop.

    3.2 Control loops (concept)

    • Inner loops
      • PID(s) holding each oven zone at its Tair,set using heater output.
      • PID(s) holding air flow at setpoint using fan speed/damper.
    • Outer loop
      • Target Ctarget (from product spec).
      • Compare Ctarget with Ĉ.
      • Controller (PID or MPC) adjusts:
        • line speed v and/or
        • oven zone setpoints Tair,set and/or
        • air-flow setpoints.
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
    k0 = st.number_input("Pre-exponential factor k0 [1/s]", value=1.0, min_value=0.0001, format="%f")
    Ea = st.number_input("Activation energy E_a [J/mol]", value=90000.0, min_value=1000.0)
    C_target = st.number_input("Target cure index C_target", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

# Derived dimensions
bare_m = d_bare_mm / 1000.0
d_overall = bare_m + 2 * (t_enamel_um / 1e6)
A_cross = math.pi * (d_overall ** 2) / 4
perimeter = math.pi * d_overall

mode = st.radio("Select mode", ["Open-loop (report cure index)", "Closed-loop (solve a setpoint)"])

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
    cure = float(k0 * math.exp(-Ea / (R_GAS * T_bar_k)) * (L_oven / v_mps) * solvent_factor)
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
        T_air = st.number_input("Zone air temperature T_air [°C]", value=420.0)
    with col_inputs2:
        v_line = st.number_input("Line speed v [m/s]", value=1.0, min_value=0.001, format="%f")
    with col_inputs3:
        h_coeff = st.number_input("Convection coefficient h [W/m²-K]", value=75.0, min_value=0.1)

    C_value, T_eff, L_theta_val = cure_index(T_air, v_line, h_coeff)

    if not math.isnan(C_value):
        st.success(
            f"Effective wire temperature ≈ {T_eff:.1f} °C (L_theta ≈ {L_theta_val:.2f} m); "
            f"cure index C ≈ {C_value:.3f}" + (" using pyrometer T" if use_pyro else "")
        )
    else:
        st.warning("Enter positive speed and convection to compute the cure index.")

else:
    st.markdown("Choose which setpoint to solve for to achieve the target cure index.")
    solve_choice = st.selectbox("Solve for", ["Line speed v", "Air temperature T_air", "Convection coefficient h"])

    # Shared knowns
    T_air_guess = st.number_input("Air temperature guess T_air [°C]", value=420.0)
    v_guess = st.number_input("Line speed guess v [m/s]", value=1.0, min_value=0.001, format="%f")
    h_guess = st.number_input("Convection guess h [W/m²-K]", value=75.0, min_value=0.1)

    def solve_for_variable(symbol: sp.Symbol, expr: sp.Expr, guess: float) -> float | None:
        guesses = []
        if guess > 0:
            guesses.extend([guess, max(guess * 0.5, 1e-6), guess * 2])
        guesses.extend([guess + 0.5, max(0.1, guess - 0.5)])
        tried = set()
        for g in guesses:
            if g in tried:
                continue
            tried.add(g)
            try:
                sol = sp.nsolve(expr - C_target, g, tol=1e-9, maxsteps=200, prec=50)
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
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_sym)
        solved = solve_for_variable(v_sym, C_expr, v_guess)
        if solved:
            st.success(f"Required line speed v ≈ {solved:.4f} m/s to hit C_target")
        else:
            st.error("Could not solve for line speed with the provided guesses.")
    elif solve_choice == "Air temperature T_air":
        T_sym = sp.symbols("Tair")
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_guess * perimeter)
        if use_pyro:
            Tbar_expr = pyro_temp_c
        else:
            Tbar_expr = T_sym - (T_sym - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_guess)
        solved = solve_for_variable(T_sym, C_expr, T_air_guess)
        if solved:
            st.success(f"Required air temperature T_air ≈ {solved:.1f} °C to hit C_target")
        else:
            st.error("Could not solve for air temperature with the provided guesses.")
    else:
        h_sym = sp.symbols("h", positive=True)
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_sym * perimeter)
        if use_pyro:
            Tbar_expr = pyro_temp_c
        else:
            Tbar_expr = T_air_guess - (T_air_guess - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = solvent_factor * k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_guess)
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

st.divider()

st.subheader("Control Loop Concept")
st.markdown(
    """
    **Open loop**: pick zone setpoints and line speed, compute the resulting cure index (C). Use
    this as a quick advisory tool during trials.

    **Closed loop**: feed temperature, line-speed, and airflow measurements into a cure estimator
    (the equations above) and adjust either line speed or setpoints to keep **C → C_target** despite
    disturbances (ambient swings, enamel solids, diameter changes). A typical architecture couples
    inner PID loops on zone temperature/airflow with an outer loop on cure index that nudges speed
    or temperature setpoints.
    """
)

st.subheader("Closed-loop block diagram")
st.caption("Illustrative flow mirroring the provided cure-control schematic.")
diagram_path = Path("data/system_diagram.png")
if diagram_path.exists():
    st.image(str(diagram_path), caption="Closed-loop cure control schematic", use_container_width=True)
else:
    st.warning("system_diagram.png not found in data/. Add the provided diagram to display it here.")
