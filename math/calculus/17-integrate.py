#!/usr/bin/env python3
"""calculates the integral
of a polynomial"""


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not poly or \
       not all(isinstance(i, (int, float)) for i in poly) or \
       not isinstance(C, (int, float)):
        return None
    result = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        result.append(int(coeff) if coeff.is_integer() else coeff)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
