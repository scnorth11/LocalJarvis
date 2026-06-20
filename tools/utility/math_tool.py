"""MathTool — safe symbolic and numeric expression evaluation via SymPy.

Uses ``sympy.sympify()`` for parsing so the agent never calls Python's
built-in ``eval()``, which would allow arbitrary code execution.  SymPy's
evaluator is restricted to mathematical operations.
"""
from __future__ import annotations

import logging
from typing import Any

from tools.base import BaseTool

logger = logging.getLogger(__name__)


class MathTool(BaseTool):
    """Evaluate mathematical expressions safely.

    Parameters (passed as ``**kwargs`` from the Executor):

    expression : str
        A mathematical expression string, e.g. ``"2**10 + sqrt(144)"``.
        Standard SymPy functions are available: ``sqrt``, ``sin``, ``cos``,
        ``log``, ``exp``, ``pi``, ``E``, ``factorial``, ``gcd``, etc.
    """

    name = "math"
    description = "Safely evaluate mathematical expressions using SymPy."

    def run(self, *, expression: str, **_: Any) -> str:
        if not expression or not expression.strip():
            return "[math] No expression provided."
        try:
            import sympy  # noqa: PLC0415  (lazy import — optional dep)
            result = sympy.sympify(expression, evaluate=True)
            # Try to return a simplified numeric answer when possible.
            numeric = sympy.nsimplify(result, rational=False)
            return str(numeric)
        except ImportError:
            return "[math] SymPy is not installed. Run: pip install sympy"
        except (sympy.SympifyError, TypeError, ValueError) as exc:
            logger.warning("MathTool failed to evaluate '%s': %s", expression, exc)
            return f"[math] Could not evaluate expression: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MathTool unexpected error for '%s': %s", expression, exc)
            return f"[math] Unexpected error: {exc}"


def create_tool(config: Any) -> MathTool:  # noqa: ARG001
    return MathTool()
