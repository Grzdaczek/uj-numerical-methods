from functools import reduce

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

	def print(self):
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

def lu_solve(vec):
	pass

def lu_det(M):
	result = 1
	for i in range(0, M.size):
		result *= M.get(i, i)

	return result

N = 100
bands = [
	[0.2		for _ in range(0, N-1)],
	[1.2		for _ in range(0, N)], # <--- diagonal
	[0.1/(n+1)		for n in range(0, N-1)],
	[0.4/(n+1)**2	for n in range(0, N-2)],
]

# bands = [
# 	[8],
# 	[4, 7],
# 	[2, 3, 19],
# 	[1, 8],
# 	[3],
# ]

A = BandMatrix(bands)
# A.print()

# print('=========================')
lu_decomp(A)
# A.print()

# print('=========================')
print('det:', lu_det(A))
