from cProfile import label
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
	"pgf.texsystem": "pdflatex",
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
})

def rnewton(x, eps, df, f):
	l = [x]
	fx = f(x)

	while True:
		x = x - (fx / df(x))
		fx = f(x)
		l.append(x)

		if np.abs(fx) < eps:
			return l

def rsecant(a, b, eps, f):
	l = [a, b]
	fa = f(a)
	fb = f(b)

	while True:
		c = (fa * b - fb * a) / (fa - fb)
		fc = f(c)
		l.append(c)
		a = b
		b = c
		fa = fb
		fb = fc

		if np.abs(fc) < eps:
			return l

def rbisect(a, b, eps, f):
	l = [a, b]
	fa = f(a)

	while True:
		c = ((a+b) / 2)
		fc = f(c)
		l.append(c)

		if np.abs(fc) < eps:
			return l

		if np.sign(fc) == np.sign(fa):
			a = c
			fa = fc
		else:
			b = c

def rfalsi(a, b, eps, f):
	l = [a, b]
	fa = f(a)
	fb = f(b)

	while True:
		c = (fa * b - fb * a) / (fa - fb)
		fc = f(c)
		l.append(c)

		if np.abs(fc) < eps:
			return l

		if np.sign(fc) == np.sign(fa):
			a = c
			fa = fc
		else:
			b = c
			fb = fc

f = lambda x: np.sin(x) - 0.37
g = lambda x: f(x)**2

df = lambda x: np.cos(x)
dg = lambda x: 2 * f(x) * df(x)

ddf = lambda x: -np.sin(x)
ddg = lambda x: 2 * df(x)**2 - f(x) * df(x)

u = lambda x: g(x) / dg(x)
du = lambda x: (dg(x)**2 - g(x)*ddg(x)) / dg(x)**2

eps = 10e-6
f0_ref = np.arcsin(0.37)

def tee(x, l = ''):
	print(l, x[-1], '\t', len(x))
	return [np.abs(x - f0_ref) for x in x]

print('---f---')
err_ne = tee(rnewton(0, eps, df, f), 'newton\t')
err_se = tee(rsecant(0, 0.5, eps, f), 'secant\t')
err_fa = tee(rfalsi(0, 3.14/2, eps, f), 'falsi\t')
err_bi = tee(rbisect(0, 3.14/2, eps, f), 'bisect\t')
fig = plt.figure(figsize=(6, 3.75))
ax = fig.gca()
ax.plot(err_ne, marker='o', label='metoda Newtona')
ax.plot(err_se, marker='o', label='metoda siecznych')
ax.plot(err_fa, marker='o', label='metoda falsi')
ax.plot(err_bi, marker='o', label='metoda bisekcji')
ax.legend()
ax.grid()
ax.set_xticks(range(len(err_bi)))
ax.set_ylabel(r'$E(i) = |x_i - x^\star|$')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('ch1.pgf')

print('---g---')
err_ne = tee(rnewton(0, eps, dg, g), 'newton\t')
err_se = tee(rsecant(0, 0.5, eps, g), 'secant\t')
fig = plt.figure(figsize=(6, 3.75))
ax = fig.gca()
ax.plot(err_ne, marker='o', label='metoda Newtona')
ax.plot(err_se, marker='o', label='metoda siecznych')
ax.legend()
ax.grid()
ax.set_xticks(range(len(err_se)))
ax.set_ylabel(r'$E(i) = |x_i - x^\star|$')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('ch2.pgf')

print('---h---')
err_ne = tee(rnewton(0, eps, du, u), 'newton\t')
err_se = tee(rsecant(0, 0.5, eps, u), 'secant\t')
fig = plt.figure(figsize=(6, 3.75))
ax = fig.gca()
ax.plot(err_ne, marker='o', label=r'metoda Newtona')
ax.plot(err_se, marker='o', label=r'metoda siecznych')
ax.legend()
ax.grid()
ax.set_xticks(range(len(err_se)))
ax.set_ylabel(r'$E(i) = |x_i - x^\star|$')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('ch3.pgf')