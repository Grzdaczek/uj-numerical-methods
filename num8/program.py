import numpy as np
from scipy import integrate
from inspect import signature

class MemFn:
	def __init__(self, f):
		self.f = f
		self.mem = {}

	def __call__(self, *argv):
		if(argv in self.mem):
			return self.mem[argv]
		else:
			x = self.f(*argv)
			self.mem[argv] = x
			return x

class CountedFn:
	def __init__(self, f=lambda: ()):
		self.f = f
		self.i = 0

	def __call__(self, *args):
		self.i += 1
		return self.f(*args)

	def collect(self):
		return self.i

epsc = lambda x: x / 10

def newton_cotes(a, b, eps, iter, f, counter=lambda: ()):
	f = MemFn(f)
	best = 0

	for i in range(iter):
		counter()
		space = np.linspace(a, b, 2**i + 1)
		simpson = lambda p, q: ((q - p) / 6) * (f(p) + 4*f((p+q)/2) + f(q))
		subvals = [simpson(p, q) for p, q in zip(space, space[1:])]
		value = sum(subvals)
		
		if np.abs(value - best) < eps:
			return value
		else:
			best = value

	return best

def romberg(a, b, eps, iter, f, counter=lambda: ()):
	f = MemFn(f)
	best = 0
	val = {}

	for j in range(1, iter+1):
		counter()
		h = (b - a) / 2**j
		s = 2 * sum([f(a + k*h) for k in range(1, 2**j)])
		val[(j, 1)] = (h/2) * (f(a) + s + f(b))

		for k in range(2, j+1):
			p = val[(j, k-1)]
			q = val[(j-1, k-1)]
			val[(j, k)] = p + (p - q) / (4**(k-1) - 1)

		if np.abs(val[(j, j)] - best) < eps:
			return val[(j, j)]
		else:
			best = val[(j, j)]

	return best

def db_newton_cotes(a, b, c, d, eps, iter, f, counter=lambda: ()):
	inner = lambda x: newton_cotes(c, d, epsc(eps), iter, lambda y: f(x, y), counter)
	return newton_cotes(a, b, eps, iter, inner, counter)

def db_romberg(a, b, c, d, eps, iter, f, counter=lambda: ()):
	inner = lambda x: romberg(c, d, epsc(eps), iter, lambda y: f(x, y), counter)
	return newton_cotes(a, b, eps, iter, inner, counter)

fn_a = lambda x: np.sin(x)
fn_b = lambda x, y: np.log(x**2 + y**3 + 1)

params = {
	'a': 0,
	'b': 1,
	'eps': 10e-10,
	'iter': 15,
}

nc_i = CountedFn()
rb_i = CountedFn()
nc_f = CountedFn(fn_a)
rb_f = CountedFn(fn_a)
nc = newton_cotes(**params, f=nc_f, counter=nc_i)
rb = romberg(**params, f=rb_f, counter=rb_i)
(ref, ea) = integrate.quad(fn_a, 0, 1)

print('--1--')
print('nc', nc, nc_i.collect(), nc_f.collect())
print('rb', rb, rb_i.collect(), rb_f.collect())
print('ref', ref, ea)

params = {
	'a': 0,
	'b': 1,
	'c': 0,
	'd': 1,
	'eps': 10e-10,
	'iter': 15,
}

nc_i = CountedFn()
rb_i = CountedFn()
nc_f = CountedFn(fn_b)
rb_f = CountedFn(fn_b)
nc = db_newton_cotes(**params, f=nc_f, counter=nc_i)
rb = db_romberg(**params, f=rb_f, counter=rb_i)
(ref, eb) = integrate.dblquad(fn_b, 0, 1, lambda x: 0, lambda x: 1)

print('--2--')
print('nc', nc, nc_i.collect(), nc_f.collect())
print('rb', rb, rb_i.collect(), rb_f.collect())
print('ref', ref, eb)