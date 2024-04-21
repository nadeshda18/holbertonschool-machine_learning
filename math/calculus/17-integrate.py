#!/usr/bin/env python3
"""calculates the integral
of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not poly or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [C]
    return [C] + [poly[i] / (i + 1) for i in range(len(poly))]
