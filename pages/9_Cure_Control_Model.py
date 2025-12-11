import math
from io import BytesIO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
    "The controller sketch and solver follow the same flow described in the earlier LLM breakdown: "
    "the moving-wire energy balance is solved to get the exponential temperature rise, the thermal "
    "length scale \(L_\theta\) shapes the average temperature, and Arrhenius kinetics integrate "
    "over residence time to form the cure index \(C\). Closed-loop control compares a target \(C\) "
    "with the estimated \(\hat{C}\) from these equations and adjusts speed or setpoints."
)

st.subheader("How these equations form the model")
st.markdown(
    """
    * **Heating (energy balance):** A moving wire segment gains heat by convection, yielding
      the first-order ODE \(\rho_{eff} A c_{p,eff} v \tfrac{dT_w}{dz} = h P (T_{air}-T_w)\).
      Solving it gives the exponential approach \(T_w(z)\) to the air temperature.
    * **Thermal length scale:** Rearranging the solution exposes \(K\) and the thermal length
      \(L_\theta = 1/K\), which quantify how quickly the wire equilibrates; they depend on
      geometry (perimeter and area), line speed, and convection.
    * **Average temperature:** Integrating \(T_w(z)\) over the heated length \(L\) produces
      \(\bar{T}_w\), the effective temperature used in the cure integral.
    * **Cure kinetics:** Arrhenius kinetics \(k(T) = k_0 e^{-E_a/(R T)}\) integrated over the
      residence time \(L/v\) give the cure index \(C\). Meeting the target is expressed as
      \(C \rightarrow C_{target}\), which you can achieve by solving for one manipulated
      variable (speed, air temperature, or convection) while holding the others.
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


def average_wire_temp(T_air_c: float, v_mps: float, h_wm2k: float) -> tuple[float, float]:
    """Return average wire temperature in °C and thermal length L_theta [m]."""
    if h_wm2k <= 0 or v_mps <= 0:
        return math.nan, math.nan
    L_theta = (rho_eff * A_cross * cp_eff * v_mps) / (h_wm2k * perimeter)
    ratio = (L_theta / L_oven) * (1 - math.exp(-L_oven / L_theta))
    T_bar = T_air_c - (T_air_c - T_in) * ratio
    return T_bar, L_theta


def cure_index(T_air_c: float, v_mps: float, h_wm2k: float) -> float:
    T_bar_c, _ = average_wire_temp(T_air_c, v_mps, h_wm2k)
    if math.isnan(T_bar_c):
        return math.nan
    T_bar_k = T_bar_c + 273.15
    return float(k0 * math.exp(-Ea / (R_GAS * T_bar_k)) * (L_oven / v_mps))


def control_loop_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    def box(x, y, w, h, text):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor="#2c3e50",
            facecolor="#ecf0f1",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
        return rect

    box(0.1, 0.9, 1.8, 0.7, "Cure Index Setpoint\nC_target")
    box(2.1, 0.95, 0.5, 0.6, "Σ")
    box(3.0, 0.9, 2.2, 0.7, "Cure Controller\n(PID / MPC)")
    box(5.6, 1.45, 2.0, 0.6, "Line speed setpoint\nv_set")
    box(5.6, 0.25, 2.0, 0.6, "Airflow / thermal setpoints\nT_air,set, h_set")
    box(8.0, 0.25, 2.4, 1.8, "Enameling Oven\n(T_air, h, L, v, d_w, t_e)")
    box(10.8, 0.2, 2.6, 1.0, "Process measurements\n(T_air, v, h, pyrometer, exhaust)")
    box(10.8, 1.3, 2.6, 0.7, "Cure Estimator\nC_hat = f(T_air, v, h, d_w, t_e)")

    def arrow(start, end, label=None):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#2c3e50"),
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.1,
                label,
                fontsize=8,
                ha="center",
            )

    arrow((1.9, 1.25), (2.1, 1.25))
    arrow((2.6, 1.25), (3.0, 1.25))
    arrow((5.2, 1.25), (5.6, 1.75), label="speed command")
    arrow((5.2, 1.25), (5.6, 0.55), label="oven setpoints")
    arrow((7.6, 1.75), (8.0, 1.75))
    arrow((7.6, 0.55), (8.0, 0.55))
    arrow((10.4, 1.65), (10.8, 1.65))
    arrow((10.4, 0.7), (10.8, 0.7))
    arrow((13.4, 1.65), (2.35, 1.55), label="feedback C_hat")
    arrow((13.4, 0.7), (2.35, 1.0), label="measurements")

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.5)
    fig.tight_layout()
    return fig


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

    T_avg, L_theta_val = average_wire_temp(T_air, v_line, h_coeff)
    C_value = cure_index(T_air, v_line, h_coeff)

    if not math.isnan(C_value):
        st.success(
            f"Average wire temperature ≈ {T_avg:.1f} °C (L_theta ≈ {L_theta_val:.2f} m); "
            f"cure index C ≈ {C_value:.3f}"
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
        try:
            sol = sp.nsolve(expr - C_target, guess, tol=1e-9, maxsteps=100)
            return float(sol)
        except Exception:
            return None

    if solve_choice == "Line speed v":
        v_sym = sp.symbols("v", positive=True)
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_sym) / (h_guess * perimeter)
        Tbar_expr = T_air_guess - (T_air_guess - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_sym)
        solved = solve_for_variable(v_sym, C_expr, v_guess)
        if solved:
            st.success(f"Required line speed v ≈ {solved:.4f} m/s to hit C_target")
        else:
            st.error("Could not solve for line speed with the provided guesses.")
    elif solve_choice == "Air temperature T_air":
        T_sym = sp.symbols("Tair")
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_guess * perimeter)
        Tbar_expr = T_sym - (T_sym - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_guess)
        solved = solve_for_variable(T_sym, C_expr, T_air_guess)
        if solved:
            st.success(f"Required air temperature T_air ≈ {solved:.1f} °C to hit C_target")
        else:
            st.error("Could not solve for air temperature with the provided guesses.")
    else:
        h_sym = sp.symbols("h", positive=True)
        L_theta_expr = (rho_eff * A_cross * cp_eff * v_guess) / (h_sym * perimeter)
        Tbar_expr = T_air_guess - (T_air_guess - T_in) * (L_theta_expr / L_oven) * (1 - sp.exp(-L_oven / L_theta_expr))
        C_expr = k0 * sp.exp(-Ea / (R_GAS * (Tbar_expr + 273.15))) * (L_oven / v_guess)
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
diagram_fig = control_loop_diagram()
buf = BytesIO()
diagram_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)
st.image(buf, use_column_width=True)
