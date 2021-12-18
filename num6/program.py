import numpy as np
import matplotlib.pyplot as plt
import operator
from numpy.polynomial import Polynomial
from functools import reduce

def polynomial_fit(x, f_x, deg):
	def phi(j):
		k_s = [k for k in range(deg) if k != j]
		mul = lambda seq: reduce(operator.mul, seq, 1)
		return mul([Polynomial([-x[k], 1]) for k in k_s]) / mul([x[j]-x[k] for k in k_s])

	return sum([phi(j)*f_x[j] for j in range(deg)])

def chart(f, dist, filename):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
		nrows=2, 
		ncols=2,
		figsize=(6, 6),
		dpi=600,
		sharex=True,
		sharey=True
	)

	def plot(ax, n):
		x = np.linspace(-1, 1, 200)
		d = dist(n)
		p = polynomial_fit(d, f(d), n)
		ax.set_title(r'$ n={} $'.format(n))
		ax.set_ylim([-0.5, 1.5])
		ax.set_xlim([-1, 1])
		ax.scatter(d, p(d), color='black', marker='.')
		ax.plot(x, f(x), color='gray', linestyle='dashed', linewidth=1)
		ax.plot(x, p(x), linewidth=1)

	plot(ax1, 3)
	plot(ax2, 7)
	plot(ax3, 13)
	plot(ax4, 25)
	
	plt.savefig(filename)

f1 = lambda x: 1.0 / (1.0 + (25 * x**2))
f2 = lambda x: 1.0 / (1.0 + x**2)

uniform = lambda n: np.linspace(-1, 1, n)
cosine =lambda n: np.cos([(np.pi*(2*i+1))/(2*n) for i in range(n)])

chart(f1, uniform, "f1_u.png")
chart(f1, cosine, "f1_c.png")
chart(f2, uniform, "f2_u.png")
chart(f2, cosine, "f2_c.png")
