from scipy.optimize import fsolve, brentq
from sympy import lambdify
from sympy.core.relational import Equality
from alpha_solve import CellFunctionInput, CellFunctionResult, MetaFunctionResult, Variable, Context
from sympy_tools import from_latex


def meta_find_roots(input_data: CellFunctionInput) -> MetaFunctionResult:
    """
    Check if we can find roots of the expression.
    Returns use_result=True only if:
    - Cell has LaTeX content
    - LaTeX does NOT contain an equals sign (=)
    - LaTeX can be parsed into a SymPy expression
    - Expression has exactly one variable
    - That variable is NOT already defined in the context
    """
    try:
        latex = input_data.cell.get('latex', '').strip()

        # Check if there's any content
        if not latex:
            return MetaFunctionResult(index=125, name='Root Finder', use_result=False)

        # Try to parse it first
        expr = from_latex(latex)

        # Check if the parsed expression is an equation
        if isinstance(expr, Equality):
            return MetaFunctionResult(index=125, name='Root Finder', use_result=False)

        # Double-check for equals sign in raw LaTeX
        if '=' in latex:
            return MetaFunctionResult(index=125, name='Root Finder', use_result=False)

        # Check if it has exactly one free symbol
        if len(expr.free_symbols) != 1:
            return MetaFunctionResult(index=125, name='Root Finder', use_result=False)

        # Check if that variable is NOT in the context
        context_var_names = {v.name for v in input_data.context.variables}
        var_name = str(list(expr.free_symbols)[0])

        if var_name in context_var_names:
            return MetaFunctionResult(index=125, name='Root Finder', use_result=False)

        # It's suitable for root finding!
        return MetaFunctionResult(index=125, name='Root Finder', use_result=True)
    except Exception as e:
        # If anything fails, don't use this root finder
        return MetaFunctionResult(index=125, name='Root Finder', use_result=False)


def find_roots(input_data: CellFunctionInput) -> CellFunctionResult:
    """
    Find roots of an expression numerically (where expression = 0).
    Searches multiple initial points to find different roots.
    """
    latex = input_data.cell.get('latex', '').strip()

    try:
        # Parse the LaTeX expression
        expr = from_latex(latex)

        # Get the single variable
        if len(expr.free_symbols) != 1:
            return CellFunctionResult(
                visible_solutions=['Root finding requires exactly one variable'],
                new_context=input_data.context
            )

        var = list(expr.free_symbols)[0]

        # Convert to numerical function
        func = lambdify(var, expr, modules=['numpy'])

        # Search for roots using multiple initial guesses
        initial_guesses = [-100, -10, -1, 0, 1, 10, 100]
        roots_found = set()

        for guess in initial_guesses:
            try:
                root = fsolve(func, guess)[0]

                # Verify it's actually a root
                if abs(func(root)) < 1e-6:
                    # Round to reasonable precision
                    root_rounded = round(root, 10)
                    roots_found.add(root_rounded)
            except:
                pass

        # Also try interval-based method
        try:
            root = brentq(func, -1000, 1000)
            root_rounded = round(root, 10)
            if abs(func(root_rounded)) < 1e-6:
                roots_found.add(root_rounded)
        except:
            pass

        # Format the solutions
        visible_solutions = []
        new_variables = list(input_data.context.variables)

        if roots_found:
            solution_strings = []
            for root in sorted(roots_found):
                visible_solutions.append(f"{var} = {root}")
                solution_strings.append(str(root))

            # Add the variable to context with all root values
            new_var = Variable.create_numerical(str(var), solution_strings)

            # Remove old variable with same name if exists
            new_variables = [v for v in new_variables if v.name != str(var)]
            new_variables.append(new_var)
        else:
            visible_solutions.append(f"No roots found for expression")

        # Create new context with updated variables
        new_context = Context(variables=new_variables)

        return CellFunctionResult(
            visible_solutions=visible_solutions,
            new_context=new_context
        )

    except Exception as e:
        # If root finding fails, return error message
        return CellFunctionResult(
            visible_solutions=[f"Error finding roots: {str(e)}"],
            new_context=input_data.context
        )

