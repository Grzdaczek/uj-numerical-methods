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

x = uniform(200)

fig = plt.figure(figsize=(6, 6), dpi=600)
ax = fig.gca()
ax.set_xlim([-1, 1])

ax.plot(x, f(x), color='black', label=r'$ f(x) $')

s1 = s(5)
ax.plot(x, [s1(i) for i in x], label=r'$ s_5(x) $')

s2 = s(7)
ax.plot(x, [s2(i) for i in x], label=r'$ s_7(x) $')

s3 = s(10)
ax.plot(x, [s3(i) for i in x], label=r'$ s_{10}(x) $')

s4 = s(30)
ax.plot(x, [s4(i) for i in x], label=r'$ s_{30}(x) $')

fig.legend()
fig.tight_layout()
fig.savefig('chart.png')