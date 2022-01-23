from operator import ne
import numpy as np

def rnewton(x, eps, df, f):
	l = []
	fx = f(x)

	while True:
		x = x - (fx / df(x))
		fx = f(x)
		l.append(x)

		if np.abs(fx) < eps:
			return (l[-1], l)

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
			return (l[-1], l)

def rbisect(a, b, eps, f):
	l = [a, b]
	fa = f(a)

	while True:
		c = ((a+b) / 2)
		fc = f(c)
		l.append(c)

		if np.abs(fc) < eps:
			return (l[-1], l)

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
			return (l[-1], l)

		if np.sign(fc) == np.sign(fa):
			a = c
			fa = fc
		else:
			b = c
			fb = fc


f = lambda x: np.sin(x) - 0.37
# g = lambda x: f(x)**2

df = lambda x: np.cos(x)
# dg = lambda x: 

print(rnewton(0, 10e-6, df, f))
print(rfalsi(0, 3.14/2, 10e-6, f))
print(rsecant(0, 3.14/2, 10e-6, f))
print(rbisect(0, 3.14/2, 10e-6, f))