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
- Magnet-wire technical data: Elektrisola enamelled-wire tables — https://www.elektrisola.com/en-us/Products/Enamelled-Wire/Technical-Data
- Magnet wire ageing methods: Thermal Accelerated Aging Methods for Magnet Wire — https://www.researchgate.net/publication/327607082_Thermal_Accelerated_Aging_Methods_for_Magnet_Wire_A_Review
- Magnetic microwires: Magnetic Microwires – Manufacture, Properties and Applications — https://www.researchgate.net/publication/262963998_Magnetic_Microwires_Manufacture_Properties_and_Applications
- Micro-scale Nd–Fe–B magnets processing: https://www.researchgate.net/publication/378413505_Manufacturing_of_micro-scale_Nd-Fe-B_magnets_by_diamond_wire_sawing_and_electropolishing_processes
- University of Minnesota magnet-wire overview slides: https://people.ece.umn.edu/~kia/Courses/EE5323/Slides/Lect_06_Wires.pdf
- Lumped-capacitance thermal refresher: https://etrl.mechanical.illinois.edu/pdf/ME320/Lecture%2012%20-%20Lumped%20Capacitance.pdf
- PID primer used for the cure controller: https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python

## What to expect on each page

- **Landing / PDFs**: quick overview plus embedded PDFs for the cited ResearchGate papers.
- **Wire Basics & Resistance**: AWG ↔ diameter/area, DC resistance with temperature correction, current density guidance, and single-missing-variable solvers.
- **AC & Skin Depth**: skin-depth/diameter comparisons, μ₀ attribution, AC resistance estimates, and plots with hoverable readouts.
- **Coil & Magnetic**: inductance/flux/force calculators, anisotropy equations in LaTeX, μ₀ sourcing, and symbolic back-solves.
- **Thermal Losses & Fill**: copper/core loss rollups, temperature-rise checks vs. thermal class, insulation life factor, and slot fill-factor calculators.
- **Ovens & Autoclaves**: enameling line residence-time and heating models, cure-time equivalence sliders, and notes on downstream VPI/autoclave usage.
- **Standards & Research**: NEMA/IEC magnet-wire standards highlights, Elektrisola technical tables, and embedded/linked research PDFs plus slide decks.
- **Thermal Aging Methods**: digitized Figure 2–5 curves (thermogravimetry, dielectric loss, DMA, capacitance drift) with logistic/polynomial fits, hover readouts, and fit coefficients.
- **Magnetic Microwires**: LaTeX equations with Sympy solvers, digitized Figures 4/6/7 overlays with Chebyshev fits, hoverable coefficients, and stress/frequency subplot groupings.
- **Cure Control Model**: energy-balance/Arrhenius derivation (with LLM walk-through), optional pyrometer/solvent toggles in open- and closed-loop solvers for speed/temperature/convection, PID helper from the Digi-Key tutorial, a live PID simulation plot driving $C \to 1$, and the uploaded control-loop block diagram (data/system_diagram.png) rendered inline.
