from qibo import set_backend, gates, Circuit


# =====================================
# LAYERED GATE ARCHITECTURES
# =====================================


# 1. SimplifiedTwoDesign


def SimplifiedTwoDesign():
    c = Circuit(4, density_matrix=True)
    c.add(gates.RY(q, theta=0) for q in range(4))
    c.add(gates.CZ(0, 1))
    c.add(gates.CZ(2, 3))
    c.add(gates.RY(1, theta=0))
    c.add(gates.RY(2, theta=0))
    c.add(gates.M(*range(4)))
    return c


# 2. BasicEntanglerLayers: 8


def BasicEntanglerLayers():
    c = Circuit(4, density_matrix=True)
    c.add(gates.RX(0, theta=0))
    c.add(gates.RY(1, theta=0))
    c.add(gates.RZ(2, theta=0))
    c.add(gates.RX(3, theta=0))
    c.add(gates.CNOT(q, q + 1) for q in range(3))
    c.add(gates.CNOT(3, 0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.M(*range(4)))
    return c


# 3. Havlicek: 3


def Havlicek():
    c = Circuit(4, density_matrix=True)
    c.add(gates.H(q) for q in range(4))
    c.add(gates.CRZ(0, 3, theta=0))
    c.add(gates.T(0))
    c.add(gates.T(2))
    c.add(gates.CRZ(0, 1, theta=0))
    c.add(gates.CZ(1, 3))
    c.add(gates.RZ(2, theta=0))
    c.add(gates.M(*range(4)))
    return c


# ====================
# Mine
# ====================

# 4. ---> 4


def Mine1():
    c = Circuit(4, density_matrix=True)
    c.add(gates.S(q) for q in range(4))
    c.add(gates.CNOT(q, q + 1) for q in range(3))
    c.add(gates.CNOT(3, 0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.M(*range(4)))
    return c


# 5. ---> 6


def Mine2():
    c = Circuit(4, density_matrix=True)
    c.add(gates.S(q) for q in range(4))
    c.add(gates.CNOT(q, q + 1) for q in range(3))
    c.add(gates.CNOT(3, 0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.H(q) for q in range(4))
    c.add(gates.CRZ(0, 1, theta=0))
    c.add(gates.CZ(1, 3))
    c.add(gates.RZ(2, theta=0))
    c.add(gates.M(*range(4)))
    return c


# 6. ---> 12


def Mine3():
    c = Circuit(4, density_matrix=True)
    c.add(gates.RY(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.S(1))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.CZ(q, q + 1) for q in range(3))
    c.add(gates.CZ(3, 0))
    c.add(gates.H(0))
    c.add(gates.CRZ(2, 3, theta=0))
    c.add(gates.S(0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.S(2))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.RX(2, theta=0))
    c.add(gates.S(1))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.H(q) for q in range(4))
    c.add(gates.CRZ(0, 1, theta=0))
    c.add(gates.CZ(1, 3))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.M(*range(4)))
    return c


# 7. basic + havlicek ---> 11


def Mine4():
    c = Circuit(4, density_matrix=True)
    c.add(gates.RX(0, theta=0))
    c.add(gates.RY(1, theta=0))
    c.add(gates.RZ(2, theta=0))
    c.add(gates.RX(3, theta=0))
    c.add(gates.CNOT(q, q + 1) for q in range(3))
    c.add(gates.CNOT(3, 0))
    c.add(gates.RY(0, theta=0))
    c.add(gates.RZ(1, theta=0))
    c.add(gates.RX(2, theta=0))
    c.add(gates.RZ(3, theta=0))
    c.add(gates.H(q) for q in range(4))
    c.add(gates.CRZ(0, 3, theta=0))
    c.add(gates.T(0))
    c.add(gates.T(2))
    c.add(gates.CRZ(0, 1, theta=0))
    c.add(gates.CZ(1, 3))
    c.add(gates.RZ(2, theta=0))
    c.add(gates.M(*range(4)))
    return c
