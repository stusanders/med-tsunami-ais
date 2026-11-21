"""The only unit conversions allowed in the code base (no inline factors)."""
KNOT_TO_M_S = 0.514444  # exact definition is 1852/3600 = 0.514444...


def knots_to_ms(x):
    return x * KNOT_TO_M_S


def ms_to_knots(x):
    return x / KNOT_TO_M_S


def deg2rad(x):
    return x * (3.141592653589793 / 180.0)


def rad2deg(x):
    return x * (180.0 / 3.141592653589793)


def km_to_m(x):
    return x * 1000.0
