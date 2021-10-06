import numpy as np
import matplotlib.pyplot as plt

def derive_a(x, h, fn):
	return ( fn(x+h) - fn(x) ) / h

def derive_b(x, h, fn):
	return ( fn(x+h) - fn(x-h) ) / 2*h

hs = np.logspace(-64, 64, num=256, base=2, endpoint=False)

x = np.float64(0.3)
err_1 = lambda h: np.abs( derive_a(x, h, np.sin) - np.cos(x))
points = [(h, err_1(h)) for h in hs]

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
line, = ax.plot(*zip(*points))
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

x = np.float32(0.3)
err_1 = lambda h: np.abs( derive_a(x, h, np.sin) - np.cos(x))
points = [(h, err_1(h)) for h in hs]

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
line, = ax.plot(*zip(*points))
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

plt.show()