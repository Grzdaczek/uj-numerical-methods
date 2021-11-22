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

A = BandMatrix(bands)
x = [n+1 for n in range(0, N)]
y_ = [0.03287133486041399, 1.339622798096375, 2.066480295894664, 2.825543605175336, 3.557571715528883, 4.284492868897645, 5.00721018451999, 5.727664002754518, 6.446615582748809, 7.164554400995276, 7.881773878242026, 8.598465868371878, 9.314759799907844, 10.030746230199036, 10.746490321152768, 11.46204012796359, 12.177431844626687, 12.892693237901542, 13.60784595684208, 14.322907124390252, 15.03789045794619, 15.752807073551208, 16.467666073000725, 17.182474979167374, 17.897240063340146, 18.611966594532937, 19.32665903159678, 20.041321172855753, 20.755956273816828, 21.47056714061568, 22.18515620483152, 22.899725583859315, 23.61427712998635, 24.328812470561147, 25.0433330410833, 25.757840112626393, 26.472334814693667, 27.186818154368854, 27.901291032443737, 28.615754257064278, 29.33020855532933, 30.04465458319117, 30.75909293394065, 31.473524145507586, 32.1879487067645, 32.902367062989086, 33.61677962061327, 34.33118675136514, 35.045588795892535, 35.75998606694211, 36.474378852156384, 37.18876741654113, 37.90315200464761, 38.61753284250725, 39.331910139350974, 40.04628408914067, 40.76065487193609, 41.47502265511775, 42.189387594482916, 42.90374983523002, 43.6181095128443, 44.33246675389621, 45.04682167676243, 45.76117439227791, 46.47552500432681, 47.18987361037867, 47.904220301975755, 48.618565165176626, 49.332908280960545, 50.047249725596565, 50.76158957098093, 51.47592788494589, 52.19026473154275, 52.904600171301595, 53.6189342614698, 54.33326705623164, 55.04759860691019, 55.761928962153874, 56.47625816810818, 57.19058626857465, 57.90491330515779, 58.61923931740096, 59.33356434291259, 60.04788841748285, 60.76221157519233, 61.47653384851288, 62.1908552684013, 62.9051758643867, 63.61949566465192, 64.33381469610926, 65.04813298447127, 65.76245055431694, 66.47676742915336, 67.19108363147355, 67.9053991828134, 68.61971410401004, 69.33402833257784, 70.0483379441879, 70.7650588638003, 71.53915685603329]

lu_decomp(A)
y = lu_solve(A, x)
print(y==y_)