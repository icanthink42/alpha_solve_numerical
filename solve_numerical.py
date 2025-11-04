import numpy as np
from scipy.optimize import fsolve, brentq
from sympy import symbols, sympify, lambdify
from sympy.core.relational import Equality
from itertools import product
from alpha_solve import CellFunctionInput, CellFunctionResult, MetaFunctionResult, Variable, Context
from sympy_tools import from_latex


def meta_solve_numerical(input_data: CellFunctionInput) -> MetaFunctionResult:
    """
    Check if the equation can be solved numerically.
    Returns use_result=True only if:
    - Cell has LaTeX content
    - LaTeX can be parsed into a SymPy expression
    - Expression is an equation (Equality type)
    - Expression has at least one variable
    - At least one variable is NOT already defined in the context
    """
    try:
        latex = input_data.cell.get('latex', '').strip()

        # Check if there's any content
        if not latex:
            return MetaFunctionResult(index=150, name='Numerical Solver', use_result=False)

        # Try to parse it
        expr = from_latex(latex)

        # Check if it's an equation (Equality type)
        if not isinstance(expr, Equality):
            return MetaFunctionResult(index=150, name='Numerical Solver', use_result=False)

        # Check if it has free symbols (variables)
        if not expr.free_symbols:
            return MetaFunctionResult(index=150, name='Numerical Solver', use_result=False)

        # Check if at least one variable is NOT in the context
        context_var_names = {v.name for v in input_data.context.variables}
        has_unsolved_variable = any(
            str(symbol) not in context_var_names
            for symbol in expr.free_symbols
        )

        if not has_unsolved_variable:
            # All variables are already defined, don't use this solver
            return MetaFunctionResult(index=150, name='Numerical Solver', use_result=False)

        # It's solvable!
        return MetaFunctionResult(index=150, name='Numerical Solver', use_result=True)
    except Exception as e:
        # If anything fails, don't use this solver
        return MetaFunctionResult(index=150, name='Numerical Solver', use_result=False)


def solve_numerical(input_data: CellFunctionInput) -> CellFunctionResult:
    """
    Solve an equation numerically using numerical methods.
    Uses initial guess of 0 for simple cases or searches in a range.
    """
    latex = input_data.cell.get('latex', '').strip()

    try:
        # Parse the LaTeX equation
        expr = from_latex(latex)

        # If it's an Eq object, extract left and right sides
        if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
            equation = expr
        else:
            # Not an equation, can't solve
            return CellFunctionResult(
                visible_solutions=['Unable to solve: not an equation'],
                new_context=input_data.context
            )

        # Get the variable to solve for
        # Prefer variables that are NOT in the context
        variables = list(equation.free_symbols)
        if not variables:
            return CellFunctionResult(
                visible_solutions=['No variables to solve for'],
                new_context=input_data.context
            )

        # Get list of variables already in context
        context_var_names = {v.name for v in input_data.context.variables}

        # Try to find a variable not in context
        var = None
        for candidate in sorted(variables, key=str):
            if str(candidate) not in context_var_names:
                var = candidate
                break

        # If all variables are in context, we can't solve
        if var is None:
            return CellFunctionResult(
                visible_solutions=['All variables already defined in context'],
                new_context=input_data.context
            )

        # Build list of substitution combinations
        context_vars_with_values = []
        for context_var in input_data.context.variables:
            var_symbol = symbols(context_var.name)
            if var_symbol != var and var_symbol in equation.free_symbols:
                context_vars_with_values.append((var_symbol, context_var.values))

        # Convert equation to function (lhs - rhs = 0)
        equation_expr = equation.lhs - equation.rhs

        all_solutions = set()  # Use set to avoid duplicates
        visible_solutions = []

        if context_vars_with_values:
            # Get all variable symbols and their value lists
            var_symbols = [v[0] for v in context_vars_with_values]
            value_lists = [v[1] for v in context_vars_with_values]

            # Generate all combinations
            for value_combo in product(*value_lists):
                # Create substitution dictionary
                subs_dict = dict(zip(var_symbols, [sympify(v) for v in value_combo]))

                # Substitute known variables
                substituted_expr = equation_expr.subs(subs_dict)

                # Convert to numerical function
                func = lambdify(var, substituted_expr, modules=['numpy'])

                # Try multiple initial guesses to find different roots
                initial_guesses = [-100, -10, -1, 0, 1, 10, 100]
                for guess in initial_guesses:
                    try:
                        solution = fsolve(func, guess)[0]

                        # Verify the solution
                        if abs(func(solution)) < 1e-6:
                            # Round to reasonable precision
                            solution_rounded = round(solution, 10)
                            all_solutions.add(solution_rounded)
                    except:
                        pass
        else:
            # No context variables to substitute, solve directly
            func = lambdify(var, equation_expr, modules=['numpy'])

            # Try multiple initial guesses to find different roots
            initial_guesses = [-100, -10, -1, 0, 1, 10, 100]
            for guess in initial_guesses:
                try:
                    solution = fsolve(func, guess)[0]

                    # Verify the solution
                    if abs(func(solution)) < 1e-6:
                        solution_rounded = round(solution, 10)
                        all_solutions.add(solution_rounded)
                except:
                    pass

            # Also try interval-based method if we haven't found solutions yet
            if not all_solutions:
                try:
                    solution = brentq(func, -1000, 1000)
                    solution_rounded = round(solution, 10)
                    if abs(func(solution_rounded)) < 1e-6:
                        all_solutions.add(solution_rounded)
                except:
                    pass

        # Format the solutions
        new_variables = list(input_data.context.variables)

        if all_solutions:
            # Convert solutions to list and format
            solution_strings = []
            for solution in sorted(all_solutions):
                visible_solutions.append(f"{var} = {solution}")
                solution_strings.append(str(solution))

            # Add or update the variable in context with all solutions
            new_var = Variable.create_numerical(str(var), solution_strings)

            # Remove old variable with same name if exists
            new_variables = [v for v in new_variables if v.name != str(var)]
            new_variables.append(new_var)
        else:
            visible_solutions.append(f"No numerical solution found for {var}")

        # Create new context with updated variables
        new_context = Context(variables=new_variables)

        return CellFunctionResult(
            visible_solutions=visible_solutions,
            new_context=new_context
        )

    except Exception as e:
        # If solving fails, return error message
        return CellFunctionResult(
            visible_solutions=[f"Error solving equation numerically: {str(e)}"],
            new_context=input_data.context
        )

