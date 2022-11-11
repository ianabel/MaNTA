import numpy as np
with open('Exact_data.plot','rt') as filedata:
		values1 = np.genfromtxt('Exact_data.plot', unpack=True)

with open('u_t_0_1.plot','rt') as filedata:
		values2 = np.genfromtxt('u_t_0_6.plot', unpack=True)



u1 = values1[2, :]

u2 = values2[2, :]

u3 = u1-u2

print(np.linalg.norm(u3))