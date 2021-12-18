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

def chart(f):
	fig, axs = plt.subplots(2, 2)
	uniform = lambda n: np.linspace(-1, 1, n)
	x = np.linspace(-1, 1, 200)

	def plot(ax, n):
		ax.spines['left'].set_position('center')
		ax.spines['bottom'].set_position('zero')
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		ax.set_ylim([-0.5, 1.5])
		# plt.xticks([-1, -0.5, 0, 0.5, 1])
		# plt.yticks([1])
		
		u = uniform(n)
		p = polynomial_fit(u, f(u), n)
		ax.plot(x, f1(x), color='gray', linestyle='dashed')
		ax.plot(x, p(x))

	plot(axs[0], 3)
	plot(axs[1], 4)
	plot(axs[2], 6)
	plot(axs[3], 10)
	
	plt.savefig("chart.png")

f1 = lambda x: 1.0 / (1.0 + (25 * x**2))
f2 = lambda x: 1.0 / (1.0 + x**2)

chart(f1, "chart.png")

# plt.legend([
# 	r'$ f(x) $',
# 	r'$ W_3(x) $',
# 	r'$ W_5(x) $',
# 	r'$ W_9(x) $',
# ])
