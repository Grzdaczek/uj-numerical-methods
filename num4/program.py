import numpy as np
import scipy as sp
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

def gen_data(N):
	# Na potrzeby wzoru Shermana-Morrisona
	data = np.array([np.full(N, 9), np.full(N, 7)])
	offsets = np.array([0, 1])
	B = sps.dia_matrix((data, offsets), shape=(N, N))
	u = np.full(N, 1)
	v = u
	# Wyraz wolny
	b = np.full(N, 5)
	return [B, b, u, v]

def solve_sh_mor(B, b, u, v):
	start = time.time()
	# Rozwiązanie z wykożystaniem wzoru Shermana-Morrisona
	LU = sps.linalg.splu(B.tocsc())
	p = LU.solve(b)
	q = LU.solve(u)
	y = p - (q * np.dot(v, p)) / (1 + np.dot(v, q))
	t = time.time() - start
	return [y, t]

def solve_force(A, b):
	# Rozwiązanie nie wykorzystujące struktury macierzy
	start = time.time()
	y = np.linalg.solve(A, b)
	t = time.time() - start
	return [y, t]

# Rozwiązanie dla N = 50
B, b, u, v = gen_data(50)
A = B.toarray() + np.outer(u, v)
y1, _ = solve_sh_mor(B, b, u, v)
y2, _ = solve_force(A, b)
assert(np.linalg.norm(y1 - y2) < 10**(-10))
print(y1)

###############################################################################
# Porównanie czasu dla N > 50

def get_dp_sh_mor(N):
	B, b, u, v = gen_data(N)
	_, t = solve_sh_mor(B, b, u, v)
	return t

def get_dp_force(N):
	B, b, u, v = gen_data(N)
	A = B.toarray() + np.outer(u, v)
	_, t = solve_force(A, b)
	return t

def plot(s, d, filename):
	fig = plt.figure(figsize=(3, 4), dpi=600)
	ax = fig.gca()
	ax.plot(s, d)
	plt.xlabel('N')
	plt.ylabel('t[s]')
	plt.ylim([0, 0.5])
	plt.grid()
	fig.savefig(filename)

s1 = np.logspace(2, 6, dtype=int, num=100)
s2 = np.logspace(2, 3.6, dtype=int, num=100)
d1 = [get_dp_sh_mor(N) for N in s1]
d2 = [get_dp_force(N) for N in s2]

plot(s1, d1, 'fast.pgf')
plot(s2, d2, 'slow.pgf')
