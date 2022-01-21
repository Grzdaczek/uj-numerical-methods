import numpy as np
from scipy import integrate

def newton_cotes(a, b, precision, f):
	def integrate(a, b, fa, fb, prev_value):
		q = (a + b) / 2
		fq = f(q)

		value = ((b - a) / 6) * (fa + 4*fq + fb)
		err = np.abs(value - prev_value)

		if err < precision:
			return value
		else:
			left = integrate(a, q, fa, fq, value / 2)
			right = integrate(q, b, fq, fb, value / 2)
			return left + right

	return integrate(a, b, f(a), f(b), 0)

def db_newton_cotes(a, b, c, d, precision, f):
	return newton_cotes(a, b, precision, lambda x: 
		newton_cotes(c, d, precision, lambda y: f(x, y)))

class MemFn:
	def __init__(self, f):
		self.f = f
		self.mem = []

	def __call__(self, *argv):
		self.mem.append(argv)
		return self.f(*argv)

	def collect(self):
		return self.mem

fn_a = lambda x: np.sin(x)
fn_b = lambda x, y: np.log(x**2 + y**3 + 1)

mem_func_a = MemFn(fn_a)
mem_func_b = MemFn(fn_b)

nc_fn_a = newton_cotes(0, 1, 10e-10, fn_a)
(ref_fn_a, ea) = integrate.quad(fn_a, 0, 1)

nc_fn_b = db_newton_cotes(0, 1, 0, 1, 10e-5, fn_b)
(ref_fn_b, eb) = integrate.dblquad(fn_b, 0, 1, lambda x: 0, lambda x: 1)

print(nc_fn_a)
print(ref_fn_a, ea)

print(nc_fn_b)
print(ref_fn_b, eb)
