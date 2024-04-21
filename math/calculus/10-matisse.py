#!/usr/bin/env python3
"""calculate the derivate of a
polynomial
"""


def poly_derivative(poly):
    """calculate the derivate of a polynomial"""
    if not poly or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]
