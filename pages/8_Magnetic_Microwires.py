import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

COLORWAY = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
import sympy as sp
import streamlit as st

from solvers import solve_missing_symbol

MU0 = 4 * math.pi * 1e-7

st.title("Magnetic Microwires – Manufacture, Properties, and Applications")

st.markdown(
    """
    Based on **"Magnetic Microwires: Manufacture, Properties and Applications"**
    (https://www.researchgate.net/publication/262963998_Magnetic_Microwires_Manufacture_Properties_and_Applications).
    Use this page to explore the governing relationships, overlayed plots, and quick calculators
    tied to the paper's figures and trends. Hover over any curve to read values dynamically.
    """
)

st.caption(
    "μ₀ = 4π × 10⁻⁷ H/m (Engineering Toolbox: https://www.engineeringtoolbox.com/permeability-d_1923.html)."
)


def _optional(value: float) -> Optional[float]:
    return value if value != 0 else None


def _elliptic_curve(x, x0, a, b, y0):
    inside = 1 - ((x - x0) / a) ** 2
    inside_clipped = np.clip(inside, 0, None)
    return y0 + b * np.sqrt(inside_clipped)


def add_series_with_fit(
    fig,
    x,
    y,
    name: str,
    color: str,
    degree: int,
    row: int = 1,
    col: int = 1,
    log_x: bool = False,
    fit_type: str = "chebyshev",
):
    x = np.array(x)
    y = np.array(y)
    x_dense = (
        np.logspace(np.log10(min(x)), np.log10(max(x)), 400)
        if log_x
        else np.linspace(min(x), max(x), 400)
    )

    if fit_type == "elliptic":
        try:
            x0_guess = float(x.mean())
            a_guess = float((x.max() - x.min()) / 2) or 1.0
            b_guess = float((y.max() - y.min()) / 2) or 1.0
            y0_guess = float(y.min())
            popt, _ = curve_fit(
                _elliptic_curve,
                x,
                y,
                p0=[x0_guess, a_guess, b_guess, y0_guess],
                maxfev=20000,
            )
            fit_y = _elliptic_curve(x_dense, *popt)
            coeffs = popt
            fit_name = f"{name} elliptic fit"
        except Exception:
            cheb = np.polynomial.Chebyshev.fit(x, y, degree, domain=[min(x), max(x)])
            coeffs = cheb.convert().coef
            fit_y = cheb(x_dense)
            fit_name = f"{name} Chebyshev fit (deg {degree})"
    else:
        cheb = np.polynomial.Chebyshev.fit(x, y, degree, domain=[min(x), max(x)])
        coeffs = cheb.convert().coef
        fit_y = cheb(x_dense)
        fit_name = f"{name} Chebyshev fit (deg {degree})"

    use_grid = bool(getattr(fig, "_grid_ref", None))

    scatter_args = dict(
        x=x,
        y=y,
        mode="markers",
        name=f"{name} points",
        marker=dict(color=color, symbol="circle"),
        hovertemplate="x: %{x:.3g}<br>y: %{y:.3g}<extra></extra>",
    )
    fit_args = dict(
        x=x_dense,
        y=fit_y,
        mode="lines",
        name=fit_name,
        line=dict(color=color),
        hovertemplate="x: %{x:.3g}<br>y: %{y:.3g}<extra></extra>",
    )

    if use_grid:
        fig.add_trace(go.Scatter(**scatter_args), row=row, col=col)
        fig.add_trace(go.Scatter(**fit_args), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(**scatter_args))
        fig.add_trace(go.Scatter(**fit_args))
    return coeffs


st.header("Magnetoelastic Anisotropy Energy")
st.latex(r"K_\sigma = \frac{3}{2}\,\lambda_s\,\sigma")
st.markdown("λₛ: saturation magnetostriction (unitless), σ: applied or internal axial stress [Pa], K_σ: magnetoelastic anisotropy energy [J/m³].")

Ks_sym, lambda_sym, sigma_sym = sp.symbols("K_sigma lambda_s sigma", real=True)
ks = _optional(st.number_input("K_σ [J/m³] (0 to solve)", value=0.0, format="%e"))
lambda_s = _optional(st.number_input("λₛ (unitless) (0 to solve)", value=30e-6, format="%e"))
sigma = _optional(st.number_input("σ [Pa] (0 to solve)", value=50e6, format="%e"))

missing, solved = solve_missing_symbol(sp.Eq(Ks_sym, 1.5 * lambda_sym * sigma_sym), {Ks_sym: ks, lambda_sym: lambda_s, sigma_sym: sigma})
if missing and solved is not None:
    st.success(f"{missing} ≈ {solved:.4e}")
else:
    st.info("Leave exactly one field at 0 to solve for it.")

st.divider()

st.header("Effective Anisotropy Field")
st.latex(r"H_k = \frac{2 K_{\text{eff}}}{\mu_0 M_s}")
st.markdown("K_eff: effective anisotropy energy [J/m³], M_s: saturation magnetization [A/m], H_k: anisotropy field [A/m].")

Hk_sym, Keff_sym, Ms_sym = sp.symbols("H_k K_eff M_s", real=True, positive=True)
Hk = _optional(st.number_input("H_k [A/m] (0 to solve)", value=0.0, format="%e"))
Keff = _optional(st.number_input("K_eff [J/m³]", value=900.0, format="%f"))
Ms = _optional(st.number_input("M_s [A/m] (0 to solve)", value=6.0e5, format="%e"))
missing, solved = solve_missing_symbol(sp.Eq(Hk_sym, 2 * Keff_sym / (MU0 * Ms_sym)), {Hk_sym: Hk, Keff_sym: Keff, Ms_sym: Ms})
if missing and solved is not None:
    st.success(f"{missing} ≈ {solved:.4e}")
else:
    st.info("Provide two and leave one as 0 to back-solve.")

st.divider()

st.header("Domain Wall Thickness")
st.latex(r"\delta = \pi \sqrt{\frac{A}{K_{\text{eff}}}}")
st.markdown("A: exchange stiffness [J/m], K_eff: effective anisotropy energy [J/m³], δ: domain-wall thickness [m].")

delta_sym, A_sym = sp.symbols("delta A", real=True, positive=True)
delta_val = _optional(st.number_input("δ [m] (0 to solve)", value=0.0, format="%e"))
A_val = _optional(st.number_input("Exchange stiffness A [J/m] (0 to solve)", value=1.2e-11, format="%e"))
Keff_dw = _optional(st.number_input("K_eff [J/m³] (domain-wall) (0 to solve)", value=900.0, format="%f"))
missing, solved = solve_missing_symbol(sp.Eq(delta_sym, math.pi * sp.sqrt(A_sym / Keff_sym)), {delta_sym: delta_val, A_sym: A_val, Keff_sym: Keff_dw})
if missing and solved is not None:
    st.success(f"{missing} ≈ {solved:.4e}")
else:
    st.info("Leave one of δ, A, or K_eff as 0 to solve it.")

st.divider()

st.header("Overlayed Curves from the Microwire Paper")
st.caption("Digitized points approximate the paper's plots; polynomial fits smooth the trend. Hover to read coordinates.")

# Figure 4 – Magnetization vs axial magnetic field for multiple alloys
st.subheader("Figure 4 – Magnetization vs axial magnetic field")
fig4 = make_subplots(rows=1, cols=1)

fe70_h = np.array([
    -89.16215218,
    -75.85262251,
    -62.14294509,
    -48.0269638,
    -31.10379217,
    -23.05158828,
    -19.85964048,
    -17.87429205,
    -14.32221128,
    -10.77936469,
    -7.63358779,
    -4.08150702,
    -0.12312238,
    2.23467126,
    6.19613396,
    8.93868505,
    11.74895346,
    16.97857671,
    26.24661413,
    34.70819995,
    41.15981285,
    48.82110318,
    56.88869737,
    66.96934253,
    78.66596897,
])
fe70_m = np.array([
    -2.65648855,
    -2.67938931,
    -2.67938931,
    -2.70229008,
    -2.61068702,
    -2.51908397,
    -2.26717557,
    -2.03816794,
    -1.46564885,
    -0.82442748,
    -0.22900763,
    0.34351145,
    0.89312977,
    1.35114504,
    1.8778626,
    2.47328244,
    2.5648855,
    2.65648855,
    2.70229008,
    2.7480916,
    2.7480916,
    2.7480916,
    2.72519084,
    2.72519084,
    2.70229008,
])

co60_h = np.array([
    -78.48101266,
    -66.4556962,
    -56.32911392,
    -44.30379747,
    -30.37974684,
    -17.72151899,
    -13.92405063,
    -8.86075949,
    -6.32911392,
    -3.79746835,
    -2.53164557,
    2.53164557,
    3.79746835,
    6.32911392,
    8.86075949,
    12.65822785,
    22.15189873,
    32.91139241,
    43.67088608,
    51.89873418,
    59.49367089,
    67.08860759,
    74.6835443,
    81.64556962,
])
co60_m = np.array([
    -2.36470588,
    -2.36470588,
    -2.4,
    -2.36470588,
    -2.4,
    -2.4,
    -2.36470588,
    -2.15294118,
    -1.8,
    -1.30588235,
    -0.77647059,
    -0.07058824,
    0.67058824,
    1.30588235,
    1.72941176,
    2.08235294,
    2.18823529,
    2.15294118,
    2.15294118,
    2.15294118,
    2.15294118,
    2.15294118,
    2.15294118,
    2.15294118,
])

co68_h = np.array([
    -94.93670886,
    -90.50632911,
    -82.91139241,
    -76.58227848,
    -70.88607595,
    -66.4556962,
    -60.75949367,
    -56.32911392,
    -49.36708861,
    -41.7721519,
    -36.07594937,
    -28.48101266,
    -21.51898734,
    -13.92405063,
    -5.06329114,
    4.43037975,
    10.12658228,
    20.25316456,
    31.64556962,
    40.50632911,
    46.83544304,
    55.06329114,
    62.02531646,
    67.08860759,
    75.3164557,
    80.37974684,
    85.44303797,
    90.50632911,
    94.93670886,
])
co68_m = np.array([
    -5.73130035,
    -5.73482451,
    -5.74086594,
    -5.66635501,
    -5.75043153,
    -5.7539557,
    -5.5198504,
    -5.1256473,
    -4.81300345,
    -3.94404488,
    -3.55084868,
    -2.92052647,
    -2.21015535,
    -1.50028769,
    -0.63233602,
    0.2351122,
    1.02603567,
    1.89298044,
    2.91800921,
    3.54732451,
    4.25819908,
    4.96756329,
    5.20066168,
    5.35572497,
    5.42872555,
    5.42469793,
    5.42067031,
    5.41664269,
    5.41311853,
])

coeffs_fe70 = add_series_with_fit(fig4, fe70_h, fe70_m, "Fe70Si10B15C5", "#1f77b4", degree=4, row=1, col=1)
coeffs_co60 = add_series_with_fit(fig4, co60_h, co60_m, "Co60Fe15Si15B10", "#2ca02c", degree=4, row=1, col=1)
coeffs_co68 = add_series_with_fit(fig4, co68_h, co68_m, "Co68.5Si14.5B14.5Y2.5", "#d62728", degree=4, row=1, col=1)

fig4.update_layout(
    title="Figure 4 – Magnetization vs axial field",
    xaxis_title="Axial magnetic field [Oe]",
    yaxis_title="Magnetization [emu × 10⁴]",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "Chebyshev fit coefficients (deg 4): Fe70Si10B15C5="
    f"{np.round(coeffs_fe70, 4)}, Co60Fe15Si15B10={np.round(coeffs_co60, 4)}, "
    f"Co68.5Si14.5B14.5Y2.5={np.round(coeffs_co68, 4)}"
)

st.subheader("Figure 6a – ΔZ/Z vs axial field across frequencies")
fig6a = make_subplots(rows=1, cols=2, subplot_titles=("ΔZ/Z vs H", "ΔZ/Z and Hm vs frequency"))

# ΔZ/Z vs H at different frequencies
delta_h_02 = np.array([
    0.25973166,
    0.53350716,
    0.87262629,
    1.26394614,
    1.75958791,
    2.13832128,
    2.66059337,
    3.05252266,
    3.4706848,
    3.8369907,
    4.30775101,
    4.84396095,
    5.22317129,
    5.65460883,
    5.95525313,
    6.16458593,
    6.3997541,
    6.64814469,
    6.94881548,
    7.13183594,
    7.30166049,
    7.56306151,
    7.82433004,
    8.1119374,
    8.26880451,
    8.49088276,
    8.73914086,
    8.97430904,
    9.22251414,
    9.62755986,
])
delta_z_02 = np.array([
    1.01078759,
    1.38253383,
    1.73629963,
    2.1253771,
    2.6383236,
    2.70797313,
    2.88360401,
    2.8644643,
    2.77423756,
    2.41796771,
    2.09682404,
    1.68670564,
    1.43688086,
    1.20461917,
    1.07931262,
    0.86558781,
    0.84700455,
    0.72188349,
    0.57882837,
    0.48943631,
    0.48883349,
    0.3991632,
    0.39823577,
    0.25522703,
    0.18367628,
    0.18288797,
    0.14650977,
    0.12792652,
    0.12704547,
    0.07236224,
])

delta_h_05 = np.array([
    1.17369455,
    1.40859775,
    1.68250574,
    2.09979345,
    2.72455911,
    2.97191628,
    3.16712574,
    3.45385868,
    3.70148083,
    3.9099392,
    4.13161998,
    4.31434897,
    4.54930516,
    5.71306303,
    6.93079696,
    8.09548225,
    9.20730439,
    10.38417868,
    11.54716811,
    13.80751187,
    15.97635864,
    18.58938842,
])
delta_z_05 = np.array([
    1.32701592,
    1.48591839,
    1.76892176,
    2.26439792,
    3.78854936,
    4.35562264,
    4.85188711,
    5.29458126,
    5.68416881,
    6.0561469,
    6.32158718,
    6.42742942,
    6.55083475,
    5.80126766,
    3.89785785,
    2.52709071,
    1.56472623,
    0.77961562,
    0.54475714,
    0.28825491,
    0.06757442,
    0.17243127,
])

delta_h_1 = np.array([
    5.43817462,
    6.38258351,
    7.29946117,
    8.22847483,
    9.18213146,
    10.18934019,
    11.11702896,
    12.90666539,
    13.72974076,
    14.63117018,
    15.49327684,
    16.53827147,
    17.24353751,
    18.11899907,
    4.5475563,
    3.88291142,
    3.43984133,
    2.9583494,
])
delta_z_1 = np.array([
    6.17496148,
    3.5980797,
    1.96196503,
    0.94700402,
    0.92587035,
    0.03487113,
    0.09266123,
    0.06351694,
    0.11968404,
    -0.1583808,
    0.10819558,
    0.05865955,
    0.04532784,
    0.09976761,
    7.72224055,
    6.65969111,
    5.93357625,
    4.69289189,
])

delta_h_2 = np.array([
    0.71737555,
    1.08296601,
    1.38329233,
    1.74904177,
    2.06256401,
    3.01232546,
    3.16619832,
    3.42595647,
    3.56663341,
    3.84024993,
    4.08612322,
    4.35966024,
    4.54119683,
    4.73428646,
    4.92708462,
    5.26668071,
    5.58025594,
    5.74003781,
    6.10872852,
    6.47617382,
    6.92401352,
    7.34509042,
    8.26318697,
    8.69806924,
    11.55938361,
    15.45395407,
])
delta_z_2 = np.array([
    0.72518744,
    0.84812905,
    0.93580538,
    0.95225556,
    0.95114265,
    3.53904918,
    5.47308716,
    6.48382838,
    8.50665559,
    8.98489326,
    10.54588661,
    11.07737,
    11.98189801,
    13.8980483,
    16.00943289,
    16.04372439,
    16.00711433,
    13.98322059,
    12.02957919,
    10.91012071,
    8.44149248,
    6.39892273,
    3.94637371,
    1.40679755,
    1.1126651,
    0.0193136,
])

add_series_with_fit(fig6a, delta_h_02, delta_z_02, "0.2 MHz", "#1f77b4", degree=4, row=1, col=1)
add_series_with_fit(fig6a, delta_h_05, delta_z_05, "0.5 MHz", "#ff7f0e", degree=4, row=1, col=1)
add_series_with_fit(fig6a, delta_h_1, delta_z_1, "1 MHz", "#2ca02c", degree=4, row=1, col=1)
add_series_with_fit(
    fig6a,
    delta_h_2,
    delta_z_2,
    "2 MHz",
    "#d62728",
    degree=4,
    row=1,
    col=1,
    fit_type="elliptic",
)

# ΔZ/Z and Hm over frequency
freq_06a_05 = np.array([0.19016708, 0.51088529, 1.00365254, 2.01284416])
vals_06a_05 = np.array([1.23318386, 5.2690583, 6.8161435, 16.41255605])
freq_06a_1 = np.array([0.18586962, 0.48141786, 0.99754689, 1.99405709])
vals_06a_1 = np.array([0.73991031, 2.95964126, 2.95964126, 3.96860987])

add_series_with_fit(fig6a, freq_06a_05, vals_06a_05, "ΔZ/Z @0.5 MHz", "#9467bd", degree=2, row=1, col=2, log_x=True)
add_series_with_fit(fig6a, freq_06a_1, vals_06a_1, "ΔZ/Z @1 MHz", "#8c564b", degree=2, row=1, col=2, log_x=True)

fig6a.update_xaxes(title_text="Field H [Oe]", row=1, col=1)
fig6a.update_yaxes(title_text="ΔZ/Z [%]", row=1, col=1)
fig6a.update_xaxes(title_text="Frequency [MHz] (log)", row=1, col=2, type="log")
fig6a.update_yaxes(title_text="ΔZ/Z or Hm (approx) [% / Oe]", row=1, col=2)
fig6a.update_layout(hovermode="x unified", height=500, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig6a, use_container_width=True)

st.subheader("Figure 6b – ΔZ/Z vs frequency under applied stress")
fig6b = make_subplots(rows=1, cols=1)

stress_levels = {
    "0 MPa": (np.array([0.99702397, 1.89455978, 2.83309304, 3.7042117, 4.56455697, 5.50403039, 6.37476373, 7.2928445, 8.14776455, 9.92357045, 9.04283434, 10.83229579]),
               np.array([6.73883955, 23.00935403, 41.72499625, 52.7395149, 51.47052065, 38.95840076, 24.00069076, 13.81043655, 6.552463, 5.48205976, 5.10071416, 4.88609408])),
    "110 MPa": (np.array([0.97655607, 1.89540747, 2.79989435, 3.65060676, 4.55241186, 5.48404028, 6.36104654, 7.0985536, 8.14798032, 9.03622235, 9.9030409, 10.82573003]),
                 np.array([5.39405397, 19.64825705, 35.85777627, 45.28306593, 44.62587379, 35.71917871, 23.38935103, 14.17386504, 5.69691104, 3.81727063, 4.38171761, 3.41931798])),
    "221 MPa": (np.array([0.15244577, 0.9626077, 1.90329871, 2.81530693, 3.70798778, 4.5603031, 5.47066217, 6.34722146, 7.09164877, 8.11308626, 9.05021697, 9.89615148, 10.83961676]),
                 np.array([0.49142991, 5.69937705, 15.85949968, 29.74692204, 37.76735561, 40.83711642, 33.76340018, 23.20578728, 14.05152774, 6.55188503, 3.32861498, 4.19826945, 3.35843831])),
    "331 MPa": (np.array([0.45046329, 0.97677185, 1.86227044, 2.78190788, 3.66759142, 4.58228143, 5.46442007, 6.33333473, 7.06390613, 8.14104466, 9.02927128, 9.90302549, 10.82573003]),
                 np.array([1.35195242, 4.53850201, 13.53659365, 24.67414317, 32.93890456, 36.19303828, 31.01329615, 23.26666695, 14.05106536, 5.69679545, 3.87826589, 4.44282846, 3.41931798])),
    "426 MPa": (np.array([0.47123944, 0.96974372, 1.88309283, 2.79624157, 3.66905562, 4.51469729, 5.45910274, 6.37622793, 7.27305476, 8.16222154, 9.05010908, 9.889185]),
                 np.array([1.47452091, 4.90505154, 13.47582958, 22.84104873, 27.13337341, 29.1641341, 24.59654086, 18.19515961, 9.77677339, 4.23048173, 3.75639096, 4.32037556])),
    "566 MPa": (np.array([0.4434968, 0.97672561, 1.91820267, 2.79004572, 3.71156349, 4.54325679, 5.45962676, 6.36272651, 7.25919885, 8.14141457, 9.95163675]),
                 np.array([1.47405854, 4.72183457, 11.76530363, 19.90761213, 23.58963743, 25.9257212, 22.51877181, 16.72826792, 9.71543135, 4.23013494, 4.1991942])),
}

for idx, (label, (freq_arr, val_arr)) in enumerate(stress_levels.items()):
    degree = 3 if idx < 3 else 2
    color = COLORWAY[idx % len(COLORWAY)]
    coeffs = add_series_with_fit(
        fig6b,
        freq_arr,
        val_arr,
        label,
        color,
        degree=degree,
        row=1,
        col=1,
    )
    fig6b.add_annotation(
        text=f"{label} fit coeffs: {np.round(coeffs, 3)}",
        xref="paper",
        yref="paper",
        x=0,
        y=1 - idx * 0.05,
        showarrow=False,
        font=dict(size=9),
    )

fig6b.update_layout(
    title="Figure 6b – ΔZ/Z vs frequency with axial stress",
    xaxis_title="Frequency [MHz]",
    yaxis_title="ΔZ/Z [%]",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig6b, use_container_width=True)

st.subheader("Figure 7 – Anisotropy field and resonance frequency vs Dm/Dt")
dm_dt = np.array([0.16069142, 0.2206146, 0.24366197, 0.64161332])
anisotropy = np.array([13.26064736, 10.37478705, 8.15672913, 7.36967632])
dm_dt_f = np.array([0.15838668, 0.2206146, 0.24519846, 0.64314981])
res_freq = np.array([8.72913118, 5.17546848, 3.79216354, 1.16865417])

fig7 = make_subplots(specs=[[{"secondary_y": True}]])
coeffs_aniso = add_series_with_fit(fig7, dm_dt, anisotropy, "Anisotropy field", "#1f77b4", degree=2)
coeffs_freq = add_series_with_fit(fig7, dm_dt_f, res_freq, "Resonance frequency", "#d62728", degree=2, row=1, col=1)
fig7.update_yaxes(title_text="H_k [Oe]", secondary_y=False)
fig7.update_yaxes(title_text="Resonance f [GHz]", secondary_y=True)
fig7.update_xaxes(title_text="D_m/D_t")
fig7.update_layout(hovermode="x unified", title="Figure 7 – Anisotropy field and resonance vs D_m/D_t", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig7, use_container_width=True)
st.caption(
    "Fits: H_k coeffs="
    f"{np.round(coeffs_aniso, 4)}, f_res coeffs={np.round(coeffs_freq, 4)}."
)
