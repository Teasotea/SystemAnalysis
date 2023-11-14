import sympy
from sympy import Eq, Symbol, latex
import numpy
from numpy import *
import scipy.sparse.linalg


class Model:
    def __init__(self, input_file, output_file, sample_size, dimensions, degrees, poly_type, lambda_multiblock):
        """
        "input_file": input_file_text, # Текст вхідного файлу
        "output_file": output_file + ".xlsx", # Назва вихідного файлу
        "sample_size": number of elements in the sample
        "dimensions": [x1_dim, x2_dim, x3_dim, y_dim], # Розмірності векторів
        "degrees": [x1_deg, x2_deg, x3_deg], # Степені поліномів (введіть нульові для перебору та пошуку найкращих)
        "weights": weight_method,  # "Ваги цільових функцій", ["Нормоване значення", "Середнє арифметичне"]
        "poly_type": poly_type, # "Вигляд поліномів", ["Чебишова", "Лежандра", "Лаґерра", "Ерміта"]
        "lambda_multiblock": lambda_option, # Визначати λ з трьох систем рівнянь
        """

        self.output_file = output_file

        """Splitting rows matrix by columns of corespioding sizes"""
        self.n1, self.n2, self.n3, self.ny = self.n = dimensions

        """Input polynomial degrees for X1, X2, X3 respectively"""
        self.d1, self.d2, self.d3 = self.d = degrees

        self.poly_type = poly_type
        self.lambda_multiblock = lambda_multiblock

        """
        Splitting initial data into rows
        by the amount of samples, in our case 36
        """
        rows = split(array(input_file), sample_size)
        self.x1, self.x2, self.x3, self.y, _ = split(rows, cumsum(self.n), axis=1)

    def solve(self):
        """Normalizing input data for polynomial input"""
        self.x1n = self.normalize(self.x1)
        self.x2n = self.normalize(self.x2)
        self.x3n = self.normalize(self.x3)
        self.yn = self.normalize(self.y)

        """Defining b accoding to the assignment"""
        self.b = mean(self.normalize(self.y), axis=1)

        """Choosing polynom to use"""
        self.polynom = self.get_polynomial(self.poly_type)

        self.l1, self.l2, self.l3 = self.get_lambda(self.lambda_multiblock)
        self.l = hstack([self.l1, self.l2, self.l3])

        self.a1, self.a2, self.a3 = self.get_a()

        self.c1, self.c2, self.c3 = self.get_c()
        self.c = array([self.c1, self.c2, self.c3]).T

        self.predict_normalized, self.predict = self.predict()

    """Normalizing input data for polynomial input"""
    def normalize(self, x):
        return (x - numpy.min(x, axis=0)) / (numpy.max(x, axis=0) - numpy.min(x, axis=0))

    """Base logic for solving equations, given by the variant"""
    def gradient(self, a, b):
        return scipy.sparse.linalg.cg(atleast_1d(a.T @ a), atleast_1d(a.T @ b), tol=1e-12)[0]

    """Defining b accoding to the assignment"""
    def get_polynomial(self, type):
        if type == "Чебишова":
            return sympy.chebyshevt
        elif type == "Лежандра":
            return sympy.legendre

    """Finding lambdas out based on the selected option"""
    def get_lambda(self, lambda_option):
        def apply_polynom(nx, deg):
            return array([[float(self.polynom(d, x)) for x in ax for d in range(deg + 1)] for ax in nx])

        """Applying all degrees of the polynomial from 0 to d# degree"""
        self.φ1 = apply_polynom(self.x1n, self.d1)
        self.φ2 = apply_polynom(self.x2n, self.d2)
        self.φ3 = apply_polynom(self.x3n, self.d3)

        F = hstack([self.φ1, self.φ2, self.φ3])

        if lambda_option:
            return [
                apply_polynom(self.x1n, self.d1),
                apply_polynom(self.x1n, self.d1),
                apply_polynom(self.x1n, self.d1),
            ]
        else:
            dims = [self.n1, self.n2, self.n3]
            degrees = add([self.d1, self.d2, self.d3], 1)

            return split(self.gradient(F, self.b), cumsum(multiply(dims, degrees)))[:-1]

    def get_a(self):
        def sum_degree(φ, l, d):
            return sum(split(φ * l, d + 1, axis=1), axis=0)

        """Calculating ψ by summing up all degrees of the polynomial with previously found λ values"""
        self.ψ1 = sum_degree(self.φ1, self.l1, self.d1)
        self.ψ2 = sum_degree(self.φ2, self.l2, self.d2)
        self.ψ3 = sum_degree(self.φ3, self.l3, self.d3)

        """
        Finding a# values for each ψ#
        Each a# has # colums coresponding to X#
        and amount of rows always equals to ny
        """
        return [
            array([self.gradient(self.ψ1, self.yn[:, i]) for i in range(self.ny)]),
            array([self.gradient(self.ψ2, self.yn[:, i]) for i in range(self.ny)]),
            array([self.gradient(self.ψ3, self.yn[:, i]) for i in range(self.ny)]),
        ]

    def get_c(self):
        """
        Calculating Ф# that correspend to X#
        Each Ф# has ny columns coresponding to Y
        """
        P1 = self.ψ1 @ self.a1.T
        P2 = self.ψ2 @ self.a2.T
        P3 = self.ψ3 @ self.a3.T
        """
        Collecting Ф# by Y, so essentially Ф is a list of matrices
        created as [P1[:, i], P2[:, i], P3[:, i]] for each i in range(ny)
        """
        P = split(stack([P1, P2, P3], axis=-1), 3, axis=1)
        P = list(map(squeeze, P))

        """
        Finding c# values for each P#
        Amount of elements in c# equals to Y#
        """
        return [self.gradient(P[i], self.yn[:, i]) for i in range(self.ny)]

    def print_phi(self):
        for i, _c in enumerate(self.c):
            phi = sympy.symbols(rf'\Phi_{{{i+1}1:{i+1}4}}(x_1)')
            rhs = dot(phi, around(_c, 5))
            print(latex(Eq(Symbol(rf'\Phi_{{{i+1}}}(x_1, x_2, x_3)'), rhs)))

    def print_phi_extended(self):
        T = [Symbol(rf"\cdot T_{{{p}}}(x_{{1{k+1}}})") for _n, _d in zip(self.n[:-1], self.d) for k in range(_n) for p in range(_d+1)]
        a = hstack([tile(self.a1, self.d1+1), tile(self.a2, self.d2+1), tile(self.a3, self.d3+1)])

        for i, _a in enumerate(a):
            phi = dot(T, around(_a * self.l, 5))
            print(Eq(Symbol(rf"\Phi_{{{i+1}}}(x_1, x_2, x_3)"), phi))