import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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


def overlay_plot(title: str, x, y, x_label: str, y_label: str, degree: int, note: str, log_x: bool = False):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(min(x), max(x), 400)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Digitized points",
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{y_label}: %{{y:.3g}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dense,
            y=poly(x_dense),
            mode="lines",
            name=f"{degree}° poly fit",
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{y_label}: %{{y:.3g}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
    )
    if log_x:
        fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(note + f" Fit coefficients: {np.round(coeffs, 4)}")


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

# Hysteresis loop (H in Oe, magnetization normalized)
h_vals = np.array([-60, -40, -20, -10, -5, 0, 5, 10, 20, 40, 60])
m_vals = np.array([-1.0, -0.9, -0.65, -0.35, -0.1, 0, 0.1, 0.4, 0.7, 0.92, 1.0])
overlay_plot("Quasi-static hysteresis (Fig. 3 style)", h_vals, m_vals, "H [Oe]", "M/Ms", degree=5, note="S-shaped magnetization mirrors the soft microwire loops.")

# Giant magnetoimpedance magnitude vs frequency
freq = np.array([1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7])
gmi = np.array([120, 135, 160, 185, 200, 175, 150])
overlay_plot("GMI ratio vs frequency (Fig. 9 style)", freq, gmi, "Frequency [Hz]", "|Z|/R_dc [%]", degree=3, note="Peak near a few MHz aligns with typical Co-based microwire GMI curves.", log_x=True)

# Time derivative of magnetization after a field step
time = np.array([0, 2, 4, 6, 8, 10, 12])
dmdt = np.array([0.9, 0.65, 0.45, 0.3, 0.2, 0.12, 0.08])
overlay_plot("dM/dt relaxation (after field jump)", time, dmdt, "Time [ms]", "dM/dt (norm.)", degree=2, note="Shows viscous relaxation seen in stress-annealed microwires.")

# Coercivity vs annealing field
anneal_field = np.array([0, 5, 10, 20, 40, 60, 80])
coercivity = np.array([0.6, 0.55, 0.48, 0.38, 0.3, 0.28, 0.27])
overlay_plot("Coercivity vs transverse annealing field", anneal_field, coercivity, "Annealing field [Oe]", "Hc [Oe]", degree=3, note="Captures stress relief and induced anisotropy reduction after field anneals.")

st.divider()

st.header("Illustrative Diagrams (referencing Figs. 1, 3, and 9)")
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Fig.1-inspired processing schematic
axes[0].axis("off")
boxes = ["Glass-coated rod", "Induction furnace", "Wire drawing", "Cooling", "Take-up spool"]
for i, label in enumerate(boxes):
    axes[0].text(0.1 + i * 0.18, 0.5, label, bbox=dict(facecolor="#e0f0ff", edgecolor="#1f77b4"), ha="center")
    if i < len(boxes) - 1:
        axes[0].annotate("", xy=(0.16 + i * 0.18, 0.5), xytext=(0.14 + i * 0.18, 0.5),
                         arrowprops=dict(arrowstyle="->", color="#1f77b4"))
axes[0].set_title("Fig. 1 style – glass-coated microwire line")

# Fig.3-inspired domain structure sketch
axes[1].axis("off")
axes[1].add_patch(plt.Circle((0.5, 0.5), 0.35, color="#ffe0e0", ec="#d62728", lw=2, alpha=0.6))
axes[1].add_patch(plt.Circle((0.5, 0.5), 0.18, color="#e0ffe0", ec="#2ca02c", lw=2, alpha=0.6))
axes[1].arrow(0.5, 0.5, 0.2, 0, head_width=0.05, color="#d62728")
axes[1].arrow(0.5, 0.5, -0.2, 0, head_width=0.05, color="#2ca02c")
axes[1].text(0.5, 0.82, "Tensile stress shell", ha="center", color="#d62728")
axes[1].text(0.5, 0.18, "Axial core", ha="center", color="#2ca02c")
axes[1].set_title("Fig. 3 style – core/shell domains")

# Fig.9-inspired GMI sensor concept
axes[2].axis("off")
axes[2].add_patch(plt.Rectangle((0.15, 0.45), 0.7, 0.1, color="#ccccff", ec="#1f77b4"))
axes[2].add_patch(plt.Rectangle((0.35, 0.4), 0.3, 0.2, color="#f5f5f5", ec="#7f7f7f"))
axes[2].text(0.5, 0.55, "Microwire", ha="center", color="#1f77b4")
axes[2].text(0.5, 0.43, "Pickup coil", ha="center", color="#7f7f7f")
axes[2].arrow(0.2, 0.65, 0.6, 0, head_width=0.03, color="#ff7f0e")
axes[2].text(0.5, 0.7, "Driving field/AC excitation", ha="center", color="#ff7f0e")
axes[2].set_title("Fig. 9 style – GMI sensing coil")

st.pyplot(fig)
st.caption("Simplified sketches inspired by the paper's figures. For exact artwork, consult the embedded PDF on the Standards & Research page.")
