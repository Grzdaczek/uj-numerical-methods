import numpy as np

class BandMatrix:
	def __init__(self, bands) -> None:
		self.bands = bands
		self.width = len(bands)
		diagonal = max(bands, key=lambda b: len(b))
		self.size = len(diagonal)
		self.diagonal_index = bands.index(diagonal)

	def pos_to_band(self, i, j) -> tuple[float, float] or None:
		band = j - i + self.diagonal_index
		index = min(i, j)
		if band < 0 or band > len(self.bands)-1:
			return None
		else:
			return [band, index]
	
	def get(self, i, j) -> float:
		ab = self.pos_to_band(i, j)
		if ab:
			a, b = ab
			return self.bands[a][b]
		else:
			return 0

	def set(self, i, j, x) -> None:
		ab = self.pos_to_band(i, j)
		if ab:
			a, b, = ab
			self.bands[a][b] = x

	def meaningful_above(self, i, j) -> int:
		ab = self.pos_to_band(i, j)
		if ab:
			a, _ = ab
			return len(bands) - a
		else:
			return 0

	def meaningful_before(self, i, j) -> int:
		ab = self.pos_to_band(i, j)
		if ab:
			a, _ = ab
			return a
		else:
			return 0

	def __str__(self):
		for i in range(0, self.size):
			line = ""
			for j in range(0, self.size):
				line += str(round(self.get(i, j), 4)) + "\t"
			print(line)

def lu_decomp(M):
	for p in range(0, M.size):
		upper_range = [[p, p + j] for j in range(0, M.size - p)]
		lower_range = [[i + p, p] for i in range(1, M.size - p)]

		for point in upper_range:
			i, j = point
			r = range(max(0, i - M.meaningful_above(i, j)), i)
			x = M.get(i, j) - sum([ M.get(i, k)*M.get(k, j) for k in r])
			M.set(i, j, x)

		for point in lower_range:
			i, j = point
			r = range(max(0, j - M.meaningful_before(i, j)), j)
			x = M.get(i, j) - sum([ M.get(i, k)*M.get(k, j) for k in r])
			x *= 1 / M.get(j, j)
			M.set(i, j, x)

def lu_det(M):
	result = 1
	for i in range(0, M.size):
		result *= M.get(i, i)

	return result

def lu_solve(M, x):
	y = [None for _ in range(M.size)]
	for i in range(0, M.size):
		r = range(max(i - M.meaningful_before(i, i), 0), i)
		y[i] = x[i] - sum([ y[k]*M.get(i, k) for k in r])

	z = [None for _ in range(M.size)]
	for i in reversed(range(0, M.size)):
		r = range(i + 1, max(M.size, M.meaningful_above(i, i)))
		z[i] = (y[i] - sum([ z[k]*M.get(i, k) for k in r])) / M.get(i, i)

	return z

N = 100

bands = [
	[0.2			for _ in range(0, N-1)],
	[1.2			for _ in range(0, N)], # <--- diagonal
	[0.1/(n+1)		for n in range(0, N-1)],
	[0.4/(n+1)**2	for n in range(0, N-2)],
]

x = [n+1 for n in range(0, N)]

A = BandMatrix(bands)

B = []
for i in range(0, A.size):
	B.append(np.array([ A.get(i, j) for j in range(0, A.size) ])) 
B = np.array(B)

lu_decomp(A)
y1 = lu_solve(A, x)
det1 = lu_det(A)

y2 = np.linalg.solve(B, x)
det2 = np.linalg.det(B)

tolerance = 6
roundarr = lambda arr: list(map(lambda x: round(x, tolerance), arr))
assert(round(det1, tolerance) == round(det2, tolerance))
assert(roundarr(y1) == roundarr(y2))

print("y:", y1)
print("det:", det1)
