import sympy as sp
from typing import Dict, Tuple, Optional


def solve_missing_symbol(equation: sp.Eq, symbol_values: Dict[sp.Symbol, Optional[float]]) -> Tuple[Optional[sp.Symbol], Optional[float]]:
    missing = [s for s, v in symbol_values.items() if v is None]
    if len(missing) != 1:
        return None, None

    missing_symbol = missing[0]
    known_values = {s: v for s, v in symbol_values.items() if v is not None}
    substituted = equation.subs(known_values)
    solutions = sp.solve(substituted, missing_symbol)
    if not solutions:
        return missing_symbol, None

    solution = solutions[0]
    try:
        numeric = float(solution.evalf())
    except Exception:
        numeric = float(solution)
    return missing_symbol, numeric
