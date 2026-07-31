import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.integrate
import equinox as eqx
from jaxtyping import Float, ArrayLike
import pickle

N = 20


A = np.ndarray((N,))
BU = np.ndarray((N - 1,))
BL = np.ndarray((N - 1,))

norms = np.ndarray((N,))


Pjm1 = 1.0

Pjm2 = 0.0


# I don't think this is right
am1 = 1.0  # np.sqrt(2) / np.pi ** (0.25)
bm1 = 0.0


pm1 = Polynomial([0])
p0 = Polynomial([am1])
Ps = [pm1, p0]

fig, ax = plt.subplots()
x = np.linspace(0, 4, 100)

A[0] = 1.0
Pdm1 = scipy.integrate.quad(lambda x: Ps[1](x) ** 2 * np.exp(-(x**2)), 0.0, np.inf)[0]
a = 1 / np.sqrt(np.pi)
b = 0.0
Pnorm = [Polynomial([np.sqrt(2) / np.pi ** (0.25)])]
norms = [np.sqrt(2) / np.pi ** (0.25)]

for i in range(1, N):
    Ps.append((Polynomial([-a, 1]) * Ps[i] - Polynomial([b]) * Ps[i - 1]))
    Pnum = scipy.integrate.quad(
        lambda x: Ps[i + 1](x) ** 2 * x * np.exp(-(x**2)), 0.0, np.inf
    )[0]
    Pden = scipy.integrate.quad(
        lambda x: Ps[i + 1](x) ** 2 * np.exp(-(x**2)), 0.0, np.inf
    )[0]

    Pnorm.append(Ps[i + 1] / Pden)
    norms.append(Pden)

    b = Pden / Pdm1
    a = Pnum / Pden

    Pdm1 = Pden
    A[i] = a
    BU[i - 1] = np.sqrt(b)
    BL[i - 1] = np.sqrt(b)

    ax.plot(x, Pnorm[i](x) * np.exp(-(x**2)))

mat = sp.diags([BL, A, BU], [-1, 0, 1])
es, evecs = scipy.linalg.eigh(mat.toarray())
print(es)

vals = np.arange(0, N)


abscissae = Ps[-1].roots()
w0 = []
for i in range(0, N - 1):
    num = scipy.integrate.quad(
        lambda x: Ps[N - 1](x) ** 2 * np.exp(-(x**2)), 0.0, np.inf
    )[0]
    den = Ps[N - 1](abscissae[i]) * Ps[N].deriv()(abscissae[i])
    w0.append(num / den)


weights = evecs[:, 0] ** 2 * (np.sqrt(np.pi) / 2.0)


def ftest(x):
    return x**2 * np.cos(x * 0.5)


fland = np.dot(ftest(abscissae), w0)
testsol = scipy.integrate.quad(lambda x: ftest(x) * np.exp(-(x**2)), 0.0, np.inf)[0]

print(fland / testsol)
with open("land.pkl", "wb") as file:
    pickle.dump(abscissae, file)
    pickle.dump(w0, file)


# for i in range(0, 5):
#     print(Ps[i].coef)
plt.show()
