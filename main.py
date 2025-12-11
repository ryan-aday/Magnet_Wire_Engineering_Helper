import streamlit as st

st.set_page_config(
    page_title="Magnet Wire Engineering Helper",
    page_icon="🧲",
    layout="wide",
)

st.title("Magnet Wire Engineering Helper 🧲")

st.markdown(
    """
    This Streamlit app is a magnet-wire design cheat sheet with calculators, quick-reference
    notes, and embedded research papers. Use the sidebar to switch between pages that cover
    geometry, resistance, AC effects, magnetic calculations, thermal checks, process ovens,
    and standards.
    """
)

st.subheader("What you can do here")
st.markdown(
    """
    * Size wires (AWG ↔ diameter), compute resistance vs. temperature, and estimate current density.
    * Explore AC skin-depth limits and see when litz or parallel strands are required.
    * Calculate inductance, flux density, and basic electromagnetic force for coils.
    * Check copper losses, thermal rise, insulation life, and slot fill factors.
    * Reference enameling oven/autoclave concepts with simple residence-time and heating models.
    * View and cite key research PDFs on magnet wire manufacturing, ageing, and microwires.
    """
)

st.info(
    "Tip: Each equation is shown in LaTeX with a mini-calculator underneath. Leave one field blank "
    "to solve for that variable (n−1 inputs → solve the last unknown)."
)

st.caption("Use the left sidebar navigation to open a calculator or reference page.")
