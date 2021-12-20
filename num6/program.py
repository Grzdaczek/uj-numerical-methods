import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from functools import reduce

matplotlib.use("pgf")
matplotlib.rcParams.update({
	"pgf.texsystem": "pdflatex",
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
})

def polynomial_fit(x, y, deg):
	def phi(j):
		mul = lambda seq: reduce(operator.mul, seq, 1)
		numerator = mul([Polynomial([-x[k], 1]) for k in range(deg) if k != j])
		denominator = mul([x[j]-x[k] for k in range(deg) if k != j])
		return numerator / denominator

	return sum([phi(j)*y[j] for j in range(deg)])

def chart(f, dist, filename):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
		nrows=2, 
		ncols=2,
		figsize=(6, 3.6),
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

	plot(ax1, 5)
	plot(ax2, 8)
	plot(ax3, 17)
	plot(ax4, 28)
	
	fig.tight_layout()
	fig.savefig(filename)

f1 = lambda x: 1.0 / (1.0 + (25 * x**2))
f2 = lambda x: 1.0 / (1.0 + x**2)

uniform = lambda n: np.linspace(-1, 1, n)
cosine =lambda n: np.cos([(np.pi*(2*i+1))/(2*n) for i in range(n)])

chart(f1, uniform, "f1_u.pgf")
chart(f1, cosine, "f1_c.pgf")
chart(f2, uniform, "f2_u.pgf")
chart(f2, cosine, "f2_c.pgf")
