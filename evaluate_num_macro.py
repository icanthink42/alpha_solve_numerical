"""
Proc macro that evaluates num() functions in LaTeX.

The num() function allows you to evaluate expressions numerically within LaTeX.
For example: num(\\pi) -> 3.141592653589793
             num(\\sqrt{2}) -> 1.414213562373095
             num(x) where x=5 in context -> 5

This proc macro runs before cell solution functions and replaces num() calls
with their numerical values.
"""

import re
from alpha_solve import ProcMacroInput, ProcMacroResult, MetaFunctionResult
from sympy_tools import from_latex
from sympy import N


def evaluate_num_functions(input_data: ProcMacroInput) -> ProcMacroResult:
    """
    Proc macro that evaluates num() functions in LaTeX.

    Finds patterns like num(...) in the LaTeX and replaces them with
    their numerical evaluation.

    Args:
        input_data: ProcMacroInput containing latex and context

    Returns:
        ProcMacroResult with num() functions replaced by their numerical values
    """
    modified_latex = input_data.latex

    # Pattern to match n(...) or n\left(...\right)
    # Match n followed by optional whitespace, then ( or \left(
    pattern = r'\\?n\s*(?:\\left)?\('

    while True:
        match = re.search(pattern, modified_latex)
        if not match:
            break

        # Check if we matched \left(
        has_left = '\\left' in match.group()

        # Find the matching closing parenthesis or \right)
        start_pos = match.end()
        paren_count = 1
        end_pos = start_pos

        # Look for closing parenthesis or \right)
        i = start_pos
        while i < len(modified_latex) and paren_count > 0:
            # Check for \left(
            if modified_latex[i:i+6] == '\\left(':
                paren_count += 1
                i += 6
            # Check for \right)
            elif modified_latex[i:i+7] == '\\right)':
                paren_count -= 1
                i += 7
            # Check for regular parentheses
            elif modified_latex[i] == '(':
                paren_count += 1
                i += 1
            elif modified_latex[i] == ')':
                paren_count -= 1
                i += 1
            else:
                i += 1

        end_pos = i

        if paren_count != 0:
            # Unmatched parentheses, skip this one
            break

        # Extract the expression inside n(...)
        expr_latex = modified_latex[start_pos:end_pos].strip()

        # Remove trailing \right) if present
        if expr_latex.endswith('\\right)'):
            expr_latex = expr_latex[:-7].strip()
        elif expr_latex.endswith(')'):
            expr_latex = expr_latex[:-1].strip()

        try:
            # Parse the LaTeX expression
            expr = from_latex(expr_latex)

            # Substitute variables from context
            subs_dict = {}
            for var in input_data.context.variables:
                if var.type == 'numerical' and len(var.values) > 0:
                    # Use the first value
                    try:
                        value = float(var.values[0])
                        subs_dict[var.name] = value
                    except (ValueError, IndexError):
                        pass
                elif var.type == 'analytical' and len(var.values) > 0:
                    # Try to convert analytical value to number
                    try:
                        from sympy.parsing.sympy_parser import parse_expr
                        analytical_expr = parse_expr(var.values[0])
                        numerical_value = N(analytical_expr)
                        subs_dict[var.name] = numerical_value
                    except:
                        pass

            # Substitute variables
            if subs_dict:
                expr = expr.subs(subs_dict)

            # Evaluate numerically
            result = N(expr)

            # Convert to string
            result_str = str(result)

            # Replace the num(...) with the result
            full_match = modified_latex[match.start():end_pos]
            modified_latex = modified_latex[:match.start()] + result_str + modified_latex[end_pos:]

        except Exception as e:
            # If evaluation fails, leave the num() as is and move on
            # We need to skip this match to avoid infinite loop
            # Replace with a marker temporarily
            marker = f"__NUM_FAILED_{match.start()}__"
            full_match = modified_latex[match.start():end_pos]
            modified_latex = modified_latex[:match.start()] + marker + modified_latex[end_pos:]
            # Then restore it
            modified_latex = modified_latex.replace(marker, full_match)
            break

    return ProcMacroResult(modified_latex=modified_latex)


def meta_evaluate_num_functions(input_data: ProcMacroInput) -> MetaFunctionResult:
    """
    Meta function that determines if evaluate_num_functions should be used.

    This runs before the proc macro to decide if it should be applied.

    Args:
        input_data: ProcMacroInput containing latex and context

    Returns:
        MetaFunctionResult indicating whether to use this proc macro
    """
    # Check if the latex contains n(...) patterns
    has_num = bool(re.search(r'\\?n\s*(?:\\left)?\(', input_data.latex))

    return MetaFunctionResult(
        index=5,  # Priority order (lower runs first)
        name="Evaluate num() Functions",
        use_result=has_num
    )


