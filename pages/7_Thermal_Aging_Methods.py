import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.title("Thermal Accelerated Aging Methods")

st.markdown(
    """
    Deeper look at accelerated ageing characterisation techniques for magnet wire, based on the
    review **""Thermal Accelerated Aging Methods for Magnet Wire – A Review""**. The digitized
    curves below are fitted to simple models so you can inspect trends and read values dynamically
    by hovering over each trace.
    """
)

st.caption(
    "Curves are approximate, digitized representations of the review's figures to illustrate how "
    "thermogravimetry, dielectric loss, dynamic mechanical analysis, and capacitance change behave "
    "through ageing. Use the linked paper for authoritative values."
)

st.markdown("Reference paper: https://www.researchgate.net/publication/327607082_Thermal_Accelerated_Aging_Methods_for_Magnet_Wire_A_Review")


def plot_with_fit(title: str, x, y, x_label: str, y_label: str, degree: int, note: str):
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
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dense,
            y=poly(x_dense),
            mode="lines",
            name=f"{degree}° poly fit",
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(note + f" Fit coefficients: {np.round(coeffs, 4)}")


st.header("Thermogravimetry (TGA)")
tga_temp = np.array([25, 100, 150, 200, 250, 300, 350, 400, 450, 500])
tga_mass = np.array([100, 99.5, 98.8, 97, 94, 86, 72, 55, 38, 22])
plot_with_fit(
    "Mass loss vs. temperature",
    tga_temp,
    tga_mass,
    "Temperature [°C]",
    "Remaining mass [%]",
    degree=3,
    note=(
        "Shows onset of enamel decomposition and volatilization. Early mass stability followed by "
        "rapid loss above ~320°C matches the review's thermogravimetric profiles."
    ),
)

st.header("Dielectric Loss Factor (tan δ)")
tan_delta_temp = np.array([60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
tan_delta = np.array([0.002, 0.003, 0.0045, 0.006, 0.012, 0.025, 0.04, 0.028, 0.015, 0.008])
plot_with_fit(
    "tan δ peak growth with temperature",
    tan_delta_temp,
    tan_delta,
    "Temperature [°C]",
    "tan δ",
    degree=4,
    note=(
        "Captures the dielectric loss peak associated with molecular mobility increase before "
        "thermal degradation. Peak location/breadth reflect the ageing-induced shift discussed in the review."
    ),
)

st.header("Dynamic Mechanical Analysis (Storage Modulus)")
dma_temp = np.array([25, 60, 80, 100, 120, 140, 160, 180, 200])
dma_modulus = np.array([3.2, 3.15, 3.1, 3.0, 2.85, 2.6, 2.3, 1.9, 1.5])  # GPa
plot_with_fit(
    "Storage modulus drop with temperature",
    dma_temp,
    dma_modulus,
    "Temperature [°C]",
    "Storage modulus [GPa]",
    degree=2,
    note=(
        "Represents stiffness reduction as bond structure relaxes. The concave drop aligns with the "
        "DMA softening trends highlighted for polyester-imide systems."
    ),
)

st.header("Capacitance Change During Thermal Ageing")
age_hours = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400])
capacitance_pct = np.array([100, 100.5, 101.5, 103, 105.5, 108, 111.5, 115])
plot_with_fit(
    "Capacitance drift vs. ageing time",
    age_hours,
    capacitance_pct,
    "Ageing time [h]",
    "Capacitance [% of initial]",
    degree=3,
    note=(
        "Illustrates moisture absorption and dielectric constant drift observed during long-term "
        "thermal exposure. Upward trend mirrors capacitance creep in the review's long-duration studies."
    ),
)

st.info(
    "Hover anywhere on the curves to read precise coordinates. Swap polynomial order or data in the "
    "code if you digitize your own points for tighter fits to specific specimens or insulation chemistries."
)
