import numpy as np
import scipy as sp
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt
import time

def calc(N):
	# Na potrzeby wzoru Shermana-Morrisona
	data = np.array([np.full(N, 9), np.full(N, 7)])
	offsets = np.array([0, 1])
	B = sps.dia_matrix((data, offsets), shape=(N, N))
	u = np.full(N, 1)
	v = u

	# Wyraz wolny
	b = np.full(N, 5)

	# Macierz A = B+uv^T
	A = B.toarray() + np.outer(u, v)

	# Sprawdzenie, nie wykorzystujące struktury macierzy
	start = time.time()
	y1 = np.linalg.solve(A, b)
	end = time.time()
	t1 = end - start

	# Rozwiązanie z wykożystaniem wzoru Shermana-Morrisona
	start = time.time()
	LU = sps.linalg.splu(B.tocsc(), permc_spec='NATURAL')
	p = LU.solve(b)
	q = LU.solve(u)
	y2 = p - (np.matmul(np.outer(q, v), p)) / (1 + np.dot(v, q))
	end = time.time()
	t2 = end - start

	tolerance = 6
	roundarr = lambda arr: list(map(lambda x: round(x, tolerance), arr))
	assert(roundarr(y1) == roundarr(y2))

	return np.array([t1, t2])

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
# 	"pgf.texsystem": "pdflatex",
# 	'font.family': 'serif',
# 	'text.usetex': True,
# 	'pgf.rcfonts': False,
# })

space = np.logspace(2, 3.5, num=50, dtype=int)
data = [calc(n) for n in space]
fig = plt.figure()
ax = fig.gca()
# ax.set_ylim(0, 2)
ax.set_ylabel('czas t[s]')
ax.set_xlabel('rozmiar macierzy N')
ax.plot(space, data)
plt.show()