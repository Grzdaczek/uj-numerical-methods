import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
	"pgf.texsystem": "pdflatex",
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
})

def derive_a(x, h, fn):
	return ( fn(x+h) - fn(x) ) / h

def derive_b(x, h, fn):
	return ( fn(x+h) - fn(x-h) ) / (2*h)

def abs_err(fn1, fn2, x):
	return np.abs( fn1(x) - fn2(x) )

def plot_err(x, h, fn, Dfn, filename):
	err_a = abs_err(lambda k: derive_a(k, h, fn), Dfn, x)
	err_b = abs_err(lambda k: derive_b(k, h, fn), Dfn, x)
	fig = plt.figure(figsize=(6, 3.75), dpi=600)
	ax = fig.gca()
	line_a, = ax.plot(h, err_a, linewidth=1)
	line_b, = ax.plot(h, err_b, linewidth=1)
	line_a.set_label('Metoda a')
	line_b.set_label('Metoda b')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend()
	plt.xlabel('log h')
	plt.ylabel('log E, dla x=0.3')
	plt.grid()
	fig.savefig(filename)

x = np.float32(0.3)
h = np.float32(np.logspace(start=-8, stop=-1, num=600, endpoint=False))
plot_err(x=0.3, h=h, fn=np.sin, Dfn=np.cos, filename='./f32.pgf')
plot_err(x=0.3, h=h, fn=np.sin, Dfn=np.cos, filename='./f32.png')

x = np.float64(0.3)
h = np.float64(np.logspace(start=-16, stop=-1, num=600, endpoint=False))
plot_err(x=0.3, h=h, fn=np.sin, Dfn=np.cos, filename='./f64.pgf')
plot_err(x=0.3, h=h, fn=np.sin, Dfn=np.cos, filename='./f64.png')
