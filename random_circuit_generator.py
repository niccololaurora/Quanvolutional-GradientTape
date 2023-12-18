import numpy as np
import random
import math
from qibo import set_backend, gates, Circuit


def couplings(length, filterdim):
    couples = []
    for _ in range(length):
        random_pair = random.sample(range(filterdim**2), 2)
        while random_pair[0] == random_pair[1]:
            random_pair = random.sample(range(filterdim**2), 2)
        couples.append(random_pair)
    return couples


def circuit_recipe_small(depth, filterdim, singleQ, twoQ):
    circuit_recipe_singleQ = []
    random_number = random.randint(1, depth)
    for _ in range(random_number):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_singleQ = random.choice(list(singleQ.keys()))
            random_value_singleQ = singleQ[random_key_singleQ]
            circuit_recipe_singleQ.append(random_value_singleQ)

    nqubits = int(filterdim**2)
    total_possible_combinations = int(
        math.factorial(nqubits) / (2 * math.factorial((nqubits - 2)))
    )
    circuit_recipe_twoQ = []
    for _ in range(total_possible_combinations):
        random_probability = random.random()
        if random_probability > 0.5:
            random_key_twoQ = random.choice(list(twoQ.keys()))
            random_value_twoQ = twoQ[random_key_twoQ]
            circuit_recipe_twoQ.append(random_value_twoQ)

    return circuit_recipe_singleQ, circuit_recipe_twoQ


def random_circuit(small_circuit, depth, filterdim, singleQ, twoQ):
    """
    Args: no
    Output: random circuit
    """

    if small_circuit == "yes":
        circuit_recipe_singleQ, circuit_recipe_twoQ = circuit_recipe_small(
            depth, filterdim, singleQ, twoQ
        )
        circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

    else:
        circuit_recipe_singleQ, circuit_recipe_twoQ = circuit_recipe(
            filterdim, singleQ, twoQ
        )
        circuit_recipe = circuit_recipe_singleQ + circuit_recipe_twoQ

    random.shuffle(circuit_recipe)

    c = Circuit(filterdim**2)
    length = len(circuit_recipe)
    couples = couplings(length, filterdim)

    for x, z in zip(circuit_recipe, couples):
        if x == "GeneralizedfSim":
            matrix = np.array(
                [[1 / 2 + 1j / 2, 1 / 2 - 1j / 2], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2]]
            )
            c.add(gates.GeneralizedfSim(z[0], z[1], unitary=matrix, phi=0))

        if x == "CU3":
            c.add(gates.CU3(z[0], z[1], theta=0, phi=0, lam=0))

        if x == "SWAP":
            c.add(gates.SWAP(z[0], z[1]))

        if x == "CNOT":
            c.add(gates.CNOT(z[0], z[1]))

        if x == "CRX":
            c.add(gates.CRX(z[0], z[1], theta=0))

        if x == "CRY":
            c.add(gates.CRY(z[0], z[1], theta=0))

        if x == "CRZ":
            c.add(gates.CRZ(z[0], z[1], theta=0))

        if x == "RX":
            c.add(gates.RX(z[0], theta=0).controlled_by(z[1]))

        if x == "RY":
            theta = random.uniform(min_angle, max_angle)
            c.add(gates.RY(z[0], theta=0).controlled_by(z[1]))

        if x == "RZ":
            c.add(gates.RZ(z[0], theta=0).controlled_by(z[1]))

        if x == "U3":
            c.add(gates.U3(z[0], theta=0, phi=0, lam=0).controlled_by(z[1]))

        if x == "S":
            c.add(gates.S(z[0]).controlled_by(z[1]))

        if x == "T":
            c.add(gates.T(z[0]).controlled_by(z[1]))

        if x == "H":
            c.add(gates.H(z[0]).controlled_by(z[1]))

    c.add(gates.M(*range(filterdim**2)))
    number_params = len(c.get_parameters())

    return number_params, circuit_recipe, c
