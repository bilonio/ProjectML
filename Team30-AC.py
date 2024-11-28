import numpy as np
from matplotlib import pyplot as plt


def fit(D, theta):
    L = np.zeros(
        len(D), np.float32
    )  # store the likelihood of each data sample for each theta
    likelihood = np.zeros(len(theta), np.float32)  # store the likelihood for each theta
    for i in range(len(theta)):

        for j in range(len(D)):
            L[j] = (1 / np.pi) * (
                1 / (1 + (D[j] - theta[i]) ** 2)
            )  # likelihood of each data sample
        likelihood[i] = np.prod(L)  # likelihood for each theta
    estimation = theta[
        np.argmax(likelihood)
    ]  # estimation is theta that maximizes the likelihood
    print(estimation)
    return estimation, likelihood, L


def predict(D, P1, P2, P1_like, P2_like):
    g_x = np.log(P1_like) - np.log(P2_like) + np.log(P1) - np.log(P2)
    return g_x


D1 = np.zeros(7, np.float32)
D1 = [2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1]
theta1 = np.linspace(-60, 60, 500)
theta1_hat, likelihood1, L1 = fit(D1, theta1)
D2 = np.zeros(5, np.float32)
D2 = [-4.5, -3.4, -3.1, -3.0, -2.3]
theta2 = np.linspace(-60, 60, 500)
theta2_hat, likelihood2, L2 = fit(D2, theta2)
# print(np.log10(likelihood1))
fig, ax = plt.subplots(2, 1)
ax[0].plot(theta1, np.log10(likelihood1))
ax[0].set_title("Likelihood of D1")
ax[0].set_xlabel("Theta")
ax[0].set_ylabel("logP(D1|θ)")
ax[1].plot(theta2, np.log10(likelihood2))
ax[1].set_xlabel("Theta")
ax[1].set_ylabel("logP(D2|θ)")
ax[1].set_title("Likelihood of D2")
plt.show()
P1 = 0.5
P2 = 0.5
# P1_like = likelihood1[np.argwhere(theta1 == theta1_hat)]
# P2_like = likelihood2[np.argwhere(theta2 == theta2_hat)]
print(L1.shape, likelihood1.shape)
g_x = predict(D1, P1, P2, P1_like, P2_like)
print(g_x)
