import numpy as np
from scipy import integrate

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

def newton_cotes(a, b, eps, iter, f):
	def integrate(a, b, fa, fb, prev_value):
		q = (a + b) / 2
		fq = f(q)

		value = ((b - a) / 6) * (fa + 4*fq + fb)
		err = np.abs(value - prev_value)

		if err < eps:
			return value
		else:
			left = integrate(a, q, fa, fq, value / 2)
			right = integrate(q, b, fq, fb, value / 2)
			return left + right

	return integrate(a, b, f(a), f(b), 0)

def db_newton_cotes(a, b, c, d, eps, iter, f):
	return newton_cotes(a, b, eps, iter, lambda x: 
		newton_cotes(c, d, eps, iter, lambda y: f(x, y)))

def romberg(a, b, eps, iter, f):
	f = MemFn(f)
	best = 0
	val = {}

	for j in range(1, iter+1):
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

def db_romberg(a, b, c, d, eps, iter, f):
	return romberg(a, b, eps, iter, lambda x: 
		romberg(c, d, eps, iter, lambda y: f(x, y)))

fn_a = lambda x: np.sin(x)
fn_b = lambda x, y: np.log(x**2 + y**3 + 1)

nc_fn_a = newton_cotes(
	a=0, 
	b=1, 
	eps=10e-6, 
	iter=15, 
	f=fn_a
)

rb_fn_a = romberg(
	a=0, 
	b=1, 
	eps=10e-10, 
	iter=15,
	f=fn_a
)

(ref_fn_a, ea) = integrate.quad(fn_a, 0, 1)

nc_fn_b = db_newton_cotes(
	a=0,
	b=1,
	c=0,
	d=1,
	eps=10e-6,
	iter=15,
	f=fn_b
)

rb_fn_b = db_romberg(
	a=0,
	b=1,
	c=0,
	d=1,
	eps=10e-10,
	iter=15,
	f=fn_b
)

(ref_fn_b, eb) = integrate.dblquad(fn_b, 0, 1, lambda x: 0, lambda x: 1)

print('nc', nc_fn_a, nc_fn_a - ref_fn_a)
print('rb', rb_fn_a, rb_fn_a - ref_fn_a)
print('ref', ref_fn_a, ea)

print()

print('nc', nc_fn_b, ref_fn_b - nc_fn_b)
print('rb', rb_fn_b, rb_fn_b - ref_fn_b)
print('ref', ref_fn_b, eb)
