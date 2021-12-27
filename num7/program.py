import math
import numpy as np
import numpy.polynomial
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

matplotlib.use('pgf')
matplotlib.rcParams.update({
	'pgf.texsystem': 'pdflatex',
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
})

class Spline:
    def __init__(self, x, f, n):
        self.n = n
        self.x = x
        self.f = f
        
        # uniform distribution
        self.h = x[1] - x[0]

        self.calc_xi()
        self.calc_p()

    def __call__(self, x):
        i = 0
        while self.x[i+1] < x:
            i += 1

        return self.p[i](x)

    def calc_xi(self):
        N = self.n - 2
        data = np.array([np.full(N, 1), np.full(N, 4), np.full(N, 1)])
        offsets = np.array([-1, 0, 1])
        A = sp.sparse.dia_matrix((data, offsets), shape=(N, N))

        b_i = lambda i: self.f[i-1] - 2*self.f[i] + self.f[i+1]
        b = np.array([b_i(i) for i in range(1, self.n-1)]) * ( 6 / self.h**2 )
        _xi = sp.sparse.linalg.spsolve(A.tocsr(), b)
        
        self.xi = np.concatenate(([0], _xi, [0]))

    def calc_p(self):
        def p(j):
            A = Polynomial([self.x[j+1], -1]) / (self.x[j+1] - self.x[j])
            B = Polynomial([-self.x[j], 1]) / (self.x[j+1] - self.x[j])
            C = 1/6 * (A**3 - A) * (self.x[j+1] - self.x[j]) ** 2
            D = 1/6 * (B**3 - B) * (self.x[j+1] - self.x[j]) ** 2
            return A*self.f[j] + B*self.f[j+1] + C*self.xi[j] + D*self.xi[j+1]

        self.p = [p(j) for j in range(self.n - 1)]

uniform = lambda n: np.linspace(-1, 1, n)
f = lambda x: 1.0 / (1.0 + (25 * x**2))
s = lambda n: Spline(uniform(n), f(uniform(n)), n)

x = uniform(5040)

s0 = s(5)
s1 = s(6)
s2 = s(7)
s3 = s(10)
s4 = s(20)

fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 8), dpi=600)

ax1.plot(x, f(x), color='black', label=r'$ f(x) $')

ax1.plot(x, [s0(i) for i in x], label=r'$ s_{5}(x) $')
ax1.plot(x, [s1(i) for i in x], label=r'$ s_{6}(x) $')
ax1.plot(x, [s2(i) for i in x], label=r'$ s_{7}(x) $')
ax1.plot(x, [s3(i) for i in x], label=r'$ s_{10}(x) $')
ax1.plot(x, [s4(i) for i in x], label=r'$ s_{20}(x) $')

ax2.plot(x, [np.abs(s0(i) - f(i)) for i in x], label=r'$ E_{5}(x) $')
ax2.plot(x, [np.abs(s1(i) - f(i)) for i in x], label=r'$ E_{6}(x) $')
ax2.plot(x, [np.abs(s2(i) - f(i)) for i in x], label=r'$ E_{7}(x) $')
ax2.plot(x, [np.abs(s3(i) - f(i)) for i in x], label=r'$ E_{10}(x) $')
ax2.plot(x, [np.abs(s4(i) - f(i)) for i in x], label=r'$ E_{20}(x) $')

ax1.grid(axis="x")
ax2.grid(axis="x")

ax1.set_ylabel(r'$ s_k(x) $, $ f(x) $')
ax2.set_ylabel(r'$ \Delta_k(x) = |s_k(x) - f(x)| $')
ax2.set_xlabel(r'$ x $')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-0.2, 1.2])
ax2.set_ylim([-0.1, 0.5])

ax1.legend()
ax2.legend()

fig.tight_layout()
fig.savefig('chart.pgf')