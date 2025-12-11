import streamlit as st

st.title("Standards & Research PDFs")

st.markdown(
    """
    Handy pointers to the major magnet-wire standards plus embedded PDFs requested for quick
    reference during design meetings. Use the direct links if your browser blocks the inline view.
    """
)

st.header("Key Standards")
st.markdown(
    """
    * **ANSI/NEMA MW 1000** – umbrella for US magnet wire types, dimensions, and test methods.
    * **IEC 60317** – international series for enamelled winding wires (round/rectangular, Cu/Al).
    * **IEC 60851** – test methods (breakdown, adhesion, flexibility, resistivity, etc.).
    * Thermal classes commonly used: 105°C, 130°C, 155°C, 180°C, 200°C, 220–240°C.
    * **Elektrisola magnet-wire technical data** – thickness builds, OD tables, resistivity, and
      insulation notes: https://www.elektrisola.com/en-us/Products/Enamelled-Wire/Technical-Data
    """
)

st.header("Additional slide decks / notes")
st.markdown(
    """
    * University of Minnesota EE5323 – magnet wire overview slides:
      https://people.ece.umn.edu/~kia/Courses/EE5323/Slides/Lect_06_Wires.pdf
    * Lumped-capacitance refresher (thermal response background):
      https://etrl.mechanical.illinois.edu/pdf/ME320/Lecture%2012%20-%20Lumped%20Capacitance.pdf
    """
)

st.header("Embedded Research PDFs")
papers = [
    (
        "Manufacturing of micro-scale Nd-Fe-B magnets by diamond wire sawing and electropolishing",
        "https://www.researchgate.net/publication/378413505_Manufacturing_of_micro-scale_Nd-Fe-B_magnets_by_diamond_wire_sawing_and_electropolishing_processes",
    ),
    (
        "Thermal Accelerated Aging Methods for Magnet Wire – A Review",
        "https://www.researchgate.net/publication/327607082_Thermal_Accelerated_Aging_Methods_for_Magnet_Wire_A_Review",
    ),
    (
        "Magnetic Microwires: Manufacture, Properties and Applications",
        "https://www.researchgate.net/publication/262963998_Magnetic_Microwires_Manufacture_Properties_and_Applications",
    ),
]

for title, url in papers:
    st.subheader(title)
    st.markdown(f"Direct link: [{url}]({url})")
    st.components.v1.iframe(url, height=500)
    st.caption("If the PDF requires login, open the direct link in a new tab or download locally.")

st.header("Figure/Graph Pointers")
st.markdown(
    """
    * Nd-Fe-B microwork (wire sawing/polishing) paper: see micrograph figures for surface finish
      quality when slicing magnets used in micro-actuators.
    * Thermal accelerated ageing review: focus on Arrhenius plots and Weibull life distributions
      for insulation; these back the Arrhenius calculator used in the Thermal page.
    * Magnetic microwires paper: diagrams of wire drawing/annealing lines and hysteresis loops help
      frame process/heat-treatment choices.
    """
)
