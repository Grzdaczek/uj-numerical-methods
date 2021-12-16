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
			def mul(t):
				j, a_ij = t
				if j != i:
					return x_last[j] * a_ij
				else:
					return 0
			row = A.getrow(i)
			s = sum(map(mul, zip(row.indices, row.data)))
			x_next[i] = (b[i] - s) / at(A, i, i)
		data.append(x_next)
		x_last = x_next
	return data

def gauss_seidel_test(A, b, n):
	x_last = b.copy()
	data = [b.copy()]
	for _ in range(0, n):
		x_next = x_last.copy()
		for i in range(0, N):
			def mul(t):
				j, a_ij = t
				if j < i:
					return x_next[j] * a_ij
				elif j > i:
					return x_last[j] * a_ij
				else:
					return 0
			row = A.getrow(i)
			s = sum(map(mul, zip(row.indices, row.data)))
			x_next[i] = (b[i] - s) / at(A, i, i)
		data.append(x_next)
		x_last = x_next
	return data

ref = sp.sparse.linalg.spsolve(A.tocsc(), x)

print("Test: gauss seidel")
y_gs = gauss_seidel_test(A, x, 50)

print("Test: jacobi")
y_j = jacobi_test(A, x, 200)

print("Calc err: gauss seidel")
err_gs = list(map(lambda v: np.linalg.norm(v-ref), y_gs))

print("Calc err: jacobi err")
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

line_b, = ax.plot(err_j, linewidth=1)
line_a, = ax.plot(err_gs, linewidth=1)

print("Plot err: gauss seidel")
line_a.set_label('metoda Gaussa-Seidela')

print("Plot err: jacobi")
line_b.set_label('metoda Jacobiego')

ax.set_yscale('log')
ax.legend()
plt.xlabel('k')
plt.ylabel('E(k)')
plt.grid()

print("Save: png")
fig.savefig('./chart.png')

print("Save: pgf")
fig.savefig('./chart.pgf')