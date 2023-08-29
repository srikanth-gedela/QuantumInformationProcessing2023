import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


class GardenOptimizationProblem:
    """Garden Optimization Problem class

    Slightly refactored version from

        https://jugit.fz-juelich.de/qip/garden-optimization-problem

    """

    def __init__(self):
        # class attributes will be set in build_garden, get_plants, and build_model
        return

    def _plot_garden(self):
        mult = 1
        fig, ax = plt.subplots(
            figsize=(mult * (self.cols - 1) + 1, mult * (self.rows - 1) + 1)
        )
        ax.axis("off")

        nodes = list(zip(*self.V))
        node_color = "w"
        for node in self.V:
            ax.text(
                *node,
                str(node),
                size=15,
                ha="center",
                va="center",
                bbox=dict(fill=True, color=node_color)
            )

        edge_color = "k--"
        for edge in self.E:
            edge = list(zip(*edge))
            ax.plot(*edge, edge_color)

        plt.title("Garden grid:")

        return fig

    def _get_garden_connectivity(self):
        n = len(self.V)
        J = np.zeros((n, n))
        for i1, n1 in enumerate(self.V):
            for i2, n2 in enumerate(self.V):
                if (n1, n2) in self.E:  # and i1 < i2:
                    J[i1, i2] = 1

        assert sum(sum(J)) == len(self.E)
        return J

    def build_garden(self, cols, rows, verbose=False):
        """Creates a planar rectangular garden grid of `cols` columns and `rows` rows."""
        self.cols = cols
        self.rows = rows

        self.G = nx.grid_2d_graph(self.cols, self.rows)
        self.V = sorted(self.G.nodes())
        self.E = sorted(self.G.edges())

        self.J = self._get_garden_connectivity()

        if verbose:
            self._plot_garden()

        return

    def pick_random_plants(self, replacement=True, seed=None):
        """Picks `n` random plants for the garden making sure that there are as many big plants as small ones.
        Set `replacement=True` to allow for the same plant species to be chosen more than once in order to create bigger problems.
        """
        plants = [
            ("Tomato", 0, 0),
            ("Paprika", 0, 0),
            ("Cucumber", 0, 0),
            ("Zuccini", 0, 0),
            ("Lettuce", 0, 0),
            ("Carrot", 0, 0),
            ("Onion", 0, 0),
            ("Raddish", 0, 0),
            ("Oregano", 0, 1),
            ("Basil", 0, 1),
            ("Thyme", 0, 1),
            ("Parsley", 0, 1),
            ("Chives", 0, 1),
            ("Rosemary", 0, 1),
            ("Sage", 0, 1),
            ("Dill", 0, 1),
            ("Coriander", 0, 1),
            ("Mint", 0, 1),
        ]

        bigs = [plant[0] for plant in plants if not plant[2]]
        smalls = [plant[0] for plant in plants if plant[2]]

        np.random.seed(seed)
        chosen_bigs = list(np.random.choice(bigs, size=self.cols, replace=replacement))
        chosen_smalls = list(
            np.random.choice(smalls, size=self.cols, replace=replacement)
        )
        chosen_plants = Counter(chosen_bigs + chosen_smalls)

        final_plants = []
        for plant in plants:
            final_plants.append((plant[0], chosen_plants[plant[0]], plant[2]))

        t, c, s = zip(*final_plants)

        return t, c, s

    def get_plants(self, t, c, s, companions="./companions.csv", verbose=False):
        # filter out unused species
        t, c, s = zip(
            *[
                (species, count, size)
                for (species, count, size) in zip(t, c, s)
                if count != 0
            ]
        )
        # make sure there are as many plant specimens as pots for each size category
        sum_big = sum([count for count, size in zip(c, s) if size == 0])
        sum_small = sum([count for count, size in zip(c, s) if size == 1])
        assert sum_small == sum_big == len(self.V) / 2

        self.t = list(t)
        self.c = {species: count for (species, count) in zip(t, c)}
        self.s = {species: smallness for (species, smallness) in zip(t, s)}

        # load companions matrix. -1: good, 0: neutral, 1: bad
        if companions.endswith('.xslx'):
            self.df = pd.read_excel(companions, index_col=0).loc[self.t, self.t]
        elif companions.endswith('.csv'): 
            self.df = pd.read_csv(companions, index_col=0).loc[self.t, self.t]
        else:
            raise ValueError('Unsupported input file format! Should be xslx or csv.')

        if verbose:
            print("Companions matrix:")
            display(self.df)

        self.C = self.df.to_numpy()

        return

    # map between (i,j) and backwards
    def _get_unary_index(self, pot, species):
        pot_idx = self.V.index(pot)
        species_idx = self.t.index(species)
        return len(self.t) * pot_idx + species_idx

    def _get_composite_index(self, unary_index):
        species = self.t[(unary_index) % len(self.t)]
        pot = self.V[unary_index // len(self.t)]
        return (pot, species)

    def build_qubo(self, l1, l2, l3, output="docplex", verbose=False):
        """Builds a QUBO instance out of the garden and plants and returns it as a BinaryQuadraticModel object (output='dwave') or QuadraticProgram (output='qiskit')."""
        # initialize QUBO
        Q = dict()
        offset = 0

        # constraint 1
        offset += l1 * len(self.V)
        for pot in self.V:
            # diagonal term
            for species in self.t:
                index = self._get_unary_index(pot, species)
                Q[(index, index)] = -1 * l1
            # off-diagonal term
            for species1 in self.t:
                for species2 in self.t:
                    if self.t.index(species1) < self.t.index(species2):
                        index1, index2 = (
                            self._get_unary_index(pot, species1),
                            self._get_unary_index(pot, species2),
                        )
                        Q[(index1, index2)] = Q.get((index1, index2), 0) + 2 * l1

        # constraint 2
        offset += l2 * sum(
            [val**2 for val in self.c.values()]
        )  # constant term c_j**2
        for species in self.t:
            # diagonal term
            for pot in self.V:
                index = self._get_unary_index(pot, species)
                Q[(index, index)] = (
                    Q.get((index, index), 0) + (1 - 2 * self.c[species]) * l2
                )
            # off-diagonal term
            for pot1 in self.V:
                for pot2 in self.V:
                    if self.V.index(pot1) < self.V.index(pot2):
                        index1, index2 = (
                            self._get_unary_index(pot1, species),
                            self._get_unary_index(pot2, species),
                        )
                        Q[(index1, index2)] = Q.get((index1, index2), 0) + 2 * l2

        # constraint 3
        offset += 0
        # diagonal term
        for pot in self.V:
            for species in self.t:
                index = self._get_unary_index(pot, species)
                Q[(index, index)] = (
                    Q.get((index, index), 0) + l3 * (pot[1] - self.s[species]) ** 2
                )

        # objective function
        offset += sum(sum(self.J))
        for pot1 in self.V:
            for pot2 in self.V:
                for species1 in self.t:
                    for species2 in self.t:
                        index1, index2 = (
                            self._get_unary_index(pot1, species1),
                            self._get_unary_index(pot2, species2),
                        )
                        Q[(index1, index2)] = (
                            Q.get((index1, index2), 0)
                            + self.J[self.V.index(pot1), self.V.index(pot2)]
                            * self.C[self.t.index(species1), self.t.index(species2)]
                        )

        self.num_vars = len(self.V) * len(self.t)
        self.offset = offset
        self.Q = np.zeros((self.num_vars, self.num_vars))
        for k, v in Q.items():
            self.Q[k[0], k[1]] = v

        if output.lower() == "docplex":
            from docplex.mp.model import Model

            mdl = Model("Garden Optimization Problem")
            [mdl.binary_var(name=f"x_{i}") for i in range(self.num_vars)]

            cost = self.offset
            for i in range(self.num_vars):
                for j in range(self.num_vars):
                    xi, xj = mdl.get_var_by_index(i), mdl.get_var_by_index(j)
                    if self.Q[i, j] != 0.0:
                        cost = cost + self.Q[i, j] * xi * xj
            mdl.minimize(cost)
            
            if verbose:
                print(mdl.prettyprint())
                
            return mdl

        elif output.lower() == "qiskit":
            from qiskit_optimization.problems import QuadraticProgram

            qp = QuadraticProgram(name="Garden Optimization Problem")
            [qp.binary_var() for _ in range(len(self.V) * len(self.t))]
            qp.minimize(quadratic=self.Q, constant=self.offset)

            if verbose:
                print("number of QUBO variables: ", qp.get_num_binary_vars())
                print("constant:\t\t\t", qp.objective.constant)
                print("linear dict:\t\t\t", qp.objective.linear.to_dict())
                print("linear array:\t\t\t", qp.objective.linear.to_array())
                print(
                    "linear array as sparse matrix:\n",
                    qp.objective.linear.coefficients,
                    "\n",
                )
                print("quadratic dict w/ index:\t", qp.objective.quadratic.to_dict())
                print(
                    "quadratic dict w/ name:\t\t",
                    qp.objective.quadratic.to_dict(use_name=True),
                )
                print(
                    "symmetric quadratic dict w/ name:\t",
                    qp.objective.quadratic.to_dict(use_name=True, symmetric=True),
                )
                print("quadratic matrix:\n", qp.objective.quadratic.to_array(), "\n")
                print(
                    "symmetric quadratic matrix:\n",
                    qp.objective.quadratic.to_array(symmetric=True),
                    "\n",
                )
                print(
                    "quadratic matrix as sparse matrix:\n",
                    qp.objective.quadratic.coefficients,
                )

            return qp

        elif output.lower() == "dwave":
            from dimod import BinaryQuadraticModel

            Q = {term: Q[term] for term in Q if Q[term] != 0}
            bqm = BinaryQuadraticModel.from_qubo(Q, offset=self.offset)

            if verbose:
                print("Number of QUBO variables: ", bqm.num_variables)
                print("Number of QUBO interactions: ", bqm.num_interactions)
                print("Energy offset:", bqm.offset)

            return bqm

    def _decode(self, sample):
        choices = []
        for q in sample:
            if sample[q]:
                choices.append(tuple(self._get_composite_index(q)))
        return choices

    def _is_invalid(self, choices, verbose=False):
        # check "fill all the pots"
        available_pots = Counter(self.V)
        chosen_pots = Counter([choice[0] for choice in choices])
        pot_dist = len(list((available_pots - chosen_pots).elements()))
        if verbose:
            print("Pot distance: ", pot_dist)
            print("Chosen pots:", chosen_pots)

        # check "place all the plants"
        available_plants = self.c
        chosen_plants = dict(Counter([choice[1] for choice in choices]))
        plant_dist = 0
        for species in available_plants:
            plant_dist += abs(available_plants[species] - chosen_plants.get(species, 0))
        if verbose:
            print("Plant distance: ", plant_dist)
            print("Chosen plants:", chosen_plants)

        # check "always look on the bright side of life"
        row_dist = 0
        for choice in choices:
            row = choice[0][1]
            row_dist += abs(row - self.s[choice[1]])

        score = pot_dist + plant_dist + row_dist

        if verbose:
            print("Row distance: ", row_dist)
            print("FINAL SCORE:", score)
            print(80 * "-")

        return score

    def evaluate_objective(self, x: np.ndarray) -> np.ndarray:
        return self.offset + x @ self.Q @ x

    def plot_solution(self, x):
        if isinstance(x, (list, np.ndarray)):
            x_dict = dict(enumerate(x))
        elif isinstance(x, dict):
            x_dict = x
            x = np.zeros(len(x_dict))
            for k, v in x_dict.items():
                x[k] = v
        else:
            raise ValueError("Wrong input type! Should be either dict or ndarray")
            
        choices = self._decode(x_dict)
        
        if self._is_invalid(choices):    
            print("Not valid solution!")
        else:
            mult = 3
            fig, ax = plt.subplots(
                figsize=(mult * (self.cols - 1) + 1, mult * (self.rows - 1) + 1)
            )
            ax.axis("off")

            chosen_pots = list(zip(*choices))[0]
            chosen_plants = list(zip(*choices))[1]

            nodes = list(zip(*self.V))
            for node in self.V:
                node_text = chosen_plants[chosen_pots.index(node)]
                ax.text(
                    *node,
                    node_text,
                    size=15,
                    ha="center",
                    va="center",
                    bbox=dict(fill=True, color="w")
                )

            for edge in self.E:
                pot_u, pot_v = edge[0], edge[1]
                plant_u, plant_v = (
                    chosen_plants[chosen_pots.index(pot_u)],
                    chosen_plants[chosen_pots.index(pot_v)],
                )

                relationship = self.df[plant_u][plant_v]
                edge_colors = {-1: "g", 0: "y", 1: "r"}
                ax.plot(*list(zip(*edge)), edge_colors[relationship])

        return
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """Is the solution feasible?"""
        x_dict = dict(enumerate(x))
        choices = self._decode(x_dict)
        return not self._is_invalid(choices)
