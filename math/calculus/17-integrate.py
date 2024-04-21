#!/usr/bin/env python3
"""calculates the integral
of a polynomial"""


def poly_integral(poly, C=0):
    """poly: a list of coefficients representing a polynomial
    C: an integer representing the integration constant
    Returns: a new list of coefficients"""
    if not isinstance(poly, list) or not poly:
        return None
    if len(poly) == 1:
        if isinstance(poly[0], (int, float)):
            if poly[0] == 0:
                return [C]
            else:
                return [C, poly[0]]
        else:
            return None
    elif not all(isinstance(i, (int, float)) for i in poly):
        return None
    result = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        result.append(int(coeff) if coeff.is_integer() else coeff)
    return result
