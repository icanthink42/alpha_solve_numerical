import numpy as np
from sympy import symbols, sympify, lambdify
from sympy.core.relational import Equality
from itertools import product
from alpha_solve import CellFunctionInput, CellFunctionResult, MetaFunctionResult
from sympy_tools import from_latex, to_latex


def meta_evaluate_numerical(input_data: CellFunctionInput) -> MetaFunctionResult:
    """
    Check if the expression can be evaluated numerically.
    Returns use_result=True only if:
    - Cell has LaTeX content
    - LaTeX does NOT contain an equals sign (=)
    - LaTeX can be parsed into a SymPy expression
    - At least one variable in the expression is defined in the context
    """
    try:
        latex = input_data.cell.get('latex', '').strip()

        # Check if there's any content
        if not latex:
            return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=False)

        # Try to parse it first
        expr = from_latex(latex)

        # Check if the parsed expression is an equation
        if isinstance(expr, Equality):
            return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=False)

        # Double-check for equals sign in raw LaTeX
        if '=' in latex:
            return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=False)

        # Check if at least one variable is in the context
        context_var_names = {v.name for v in input_data.context.variables}
        has_context_variable = any(
            str(symbol) in context_var_names
            for symbol in expr.free_symbols
        )

        if not has_context_variable and expr.free_symbols:
            # Has variables but none are in context
            return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=False)

        # It's evaluable!
        return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=True)
    except Exception as e:
        # If anything fails, don't use this evaluator
        return MetaFunctionResult(index=75, name='Numerical Evaluation', use_result=False)


def evaluate_numerical(input_data: CellFunctionInput) -> CellFunctionResult:
    """
    Evaluate an expression numerically by substituting context variables.
    Generates one solution for each combination of context variable values.
    """
    latex = input_data.cell.get('latex', '').strip()

    try:
        # Parse the LaTeX expression
        expr = from_latex(latex)

        # Build list of context variables and their values
        context_vars_with_values = []
        for context_var in input_data.context.variables:
            var_symbol = symbols(context_var.name)
            if var_symbol in expr.free_symbols and context_var.values:
                context_vars_with_values.append((var_symbol, context_var.values))

        visible_solutions = []

        if context_vars_with_values:
            # Get all variable symbols and their value lists
            var_symbols = [v[0] for v in context_vars_with_values]
            value_lists = [v[1] for v in context_vars_with_values]

            # Generate all combinations
            for value_combo in product(*value_lists):
                # Create substitution dictionary
                subs_dict = dict(zip(var_symbols, [float(sympify(v)) for v in value_combo]))

                # Convert to numerical function
                func = lambdify(var_symbols, expr, modules=['numpy'])

                # Evaluate
                if len(var_symbols) == 1:
                    result = func(subs_dict[var_symbols[0]])
                else:
                    result = func(*[subs_dict[sym] for sym in var_symbols])

                # Format result
                if isinstance(result, (np.ndarray, list)):
                    result = result.item() if hasattr(result, 'item') else result[0]

                # Round to reasonable precision
                result_rounded = round(float(result), 10)
                visible_solutions.append(str(result_rounded))
        else:
            # No context variables to substitute, just evaluate if possible
            try:
                func = lambdify([], expr, modules=['numpy'])
                result = func()
                result_rounded = round(float(result), 10)
                visible_solutions.append(str(result_rounded))
            except:
                visible_solutions.append(to_latex(expr))

        # Remove duplicates while preserving order
        visible_solutions = list(dict.fromkeys(visible_solutions))

        # Context remains unchanged
        return CellFunctionResult(
            visible_solutions=visible_solutions,
            new_context=input_data.context
        )

    except Exception as e:
        # If evaluation fails, return error message
        return CellFunctionResult(
            visible_solutions=[f"Error evaluating expression: {str(e)}"],
            new_context=input_data.context
        )

