import numpy as np
import scipy as sp
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg

N = 50

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
y1 = np.linalg.solve(A, b)

# Rozwiązanie z wykożystaniem wzoru Shermana-Morrisona
LU = sps.linalg.splu(B.tocsc())
p = LU.solve(b)
q = LU.solve(u)
y2 = p - (np.matmul(np.outer(q, v), p)) / (1 + np.dot(v, q))

tolerance = 9
roundarr = lambda arr: list(map(lambda x: round(x, tolerance), arr))
assert(roundarr(y1) == roundarr(y2))

print(y2)
