# Magnet_Wire_Engineering_Helper
Magnet wire engineering helper with various calculators and references to existing research papers.

## Running the app

```bash
pip install -r requirements.txt
streamlit run main.py
```

Navigate via the Streamlit sidebar to access calculators for wire sizing, resistance, AC effects,
magnetic and thermal checks, enameling oven/autoclave references, ageing-method visualizations with
digitized thermogravimetry/dielectric/DMA/capacitance curves, microwire property calculators/plots,
and embedded research PDFs.

## Reference links

- μ₀ source: Engineering Toolbox permeability table — https://www.engineeringtoolbox.com/permeability-d_1923.html
- Magnet wire ageing methods: Thermal Accelerated Aging Methods for Magnet Wire — https://www.researchgate.net/publication/327607082_Thermal_Accelerated_Aging_Methods_for_Magnet_Wire_A_Review
- Magnetic microwires: Magnetic Microwires – Manufacture, Properties and Applications — https://www.researchgate.net/publication/262963998_Magnetic_Microwires_Manufacture_Properties_and_Applications
- Micro-scale Nd–Fe–B magnets processing: https://www.researchgate.net/publication/378413505_Manufacturing_of_micro-scale_Nd-Fe-B_magnets_by_diamond_wire_sawing_and_electropolishing_processes

## What to expect on each page

- **Landing / PDFs**: quick overview plus embedded PDFs for the cited ResearchGate papers.
- **Wire Basics & Resistance**: AWG ↔ diameter/area, DC resistance with temperature correction, current density guidance, and single-missing-variable solvers.
- **AC & Skin Depth**: skin-depth/diameter comparisons, μ₀ attribution, AC resistance estimates, and plots with hoverable readouts.
- **Coil & Magnetic**: inductance/flux/force calculators, anisotropy equations in LaTeX, μ₀ sourcing, and symbolic back-solves.
- **Thermal Losses & Fill**: copper/core loss rollups, temperature-rise checks vs. thermal class, insulation life factor, and slot fill-factor calculators.
- **Ovens & Autoclaves**: enameling line residence-time and heating models, cure-time equivalence sliders, and notes on downstream VPI/autoclave usage.
- **Standards & Research**: NEMA/IEC magnet-wire standards highlights and links to research PDFs.
- **Thermal Aging Methods**: digitized Figure 2–5 curves (thermogravimetry, dielectric loss, DMA, capacitance drift) with logistic/polynomial fits, hover readouts, and fit coefficients.
- **Magnetic Microwires**: LaTeX equations with Sympy solvers, digitized Figures 4/6/7 overlays with Chebyshev fits, hoverable coefficients, and stress/frequency subplot groupings.
