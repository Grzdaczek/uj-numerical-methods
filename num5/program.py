import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt

N = 50

x = np.array([float(x) for x in range(1, N+1)])

data = np.array(list(map(lambda x: np.full(N, x), [0.2, 1, 3, 1, 0.2])))
offsets = np.array([-2, -1, 0, 1, 2])
A = sp.sparse.dia_matrix((data, offsets), shape=(N, N))

def at(A, i, j):
	o = list(A.offsets)
	if (i-j in o):
		d = o.index(i-j)
		return A.data[d][min(i, j) - i+j]
	else:
		return 0.0

def jacobi_test(A, b, n):
	x_last = b.copy()
	data = [b.copy()]
	for _ in range(0, n):
		x_next = x_last.copy()
		for i in range(0, N):
			s1 = sum([at(A, i, j) * x_last[j] for j in range(0, i) if at(A, i, j) != 0])
			s2 = sum([at(A, i, j) * x_last[j] for j in range(i+1, N) if at(A, i, j) != 0])
			x_next[i] = (b[i] - s1 - s2) / at(A, i, i)
		data.append(x_next)
		x_last = x_next
	return data

def gauss_seidel_test(A, b, n):
	x = b.copy()
	data = [b.copy()]
	for _ in range(0, n):
		for i in range(0, N):
			s1 = sum([at(A, i, j) * x[j] for j in range(0, i) if at(A, i, j) != 0])
			s2 = sum([at(A, i, j) * x[j] for j in range(i+1, N) if at(A, i, j) != 0])
			x[i] = (b[i] - s1 - s2) / at(A, i, i)
		data.append(x.copy())
	return data

ref = sp.sparse.linalg.spsolve(A.tocsc(), x)
y_gs = gauss_seidel_test(A, x, 20)
y_j = jacobi_test(A, x, 100)

err_gs = list(map(lambda v: np.linalg.norm(v-ref), y_gs))
err_j = list(map(lambda v: np.linalg.norm(v-ref), y_j))

matplotlib.use("pgf")
matplotlib.rcParams.update({
	"pgf.texsystem": "pdflatex",
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
})

fig = plt.figure(figsize=(6, 3.75), dpi=600)
ax = fig.gca()
line_a, = ax.plot(err_gs, linewidth=1)
line_b, = ax.plot(err_j, linewidth=1)
line_a.set_label('metoda Gaussa-Seidela')
line_b.set_label('metoda Jacobiego')
ax.set_yscale('log')
ax.legend()
plt.xlabel('k')
plt.ylabel('E(k)')
plt.grid()
fig.savefig('./chart.png')
fig.savefig('./chart.pgf')