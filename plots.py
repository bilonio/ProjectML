import numpy as np
import matplotlib.pyplot as plt

D1 = np.array([ 2.8, -0.4, -0.8, 2.3,-0.3,3.6,4.1])
D2=np.array([-4.5,-3.4,-3.1,-3.0,-2.3])

D = np.concatenate((D1,D2))

colors = [1,1,1,1,1,1,1,2,2,2,2,2]

print(np.mean(D1))
print(np.mean(D2))


plt.figure(figsize=(8, 2), dpi=80)
plt.scatter(D,np.zeros_like(D), c=colors, s=[80]*len(D))
plt.savefig('./plots/Figure_1.pdf')
plt.show()

