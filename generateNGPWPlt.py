import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

price_list = [[0.24, 0.48, 0.72], [0.24, 0.48, 0.6], [0.24, 0.48, 0.96]]
pk = 0
# x_range = [0, sum(price_list1)]
mu = np.mean(price_list[pk])
sigma = mu / 4
x_min, x_max = float(mu - 4 * sigma), float(mu + 4 * sigma)
# print(type(x_min))
X = np.arange(0, 5, 0.001)
print(mu, sigma)
print(X)

# y = (stats.norm.pdf(X, 0, 1)) * 2
# y = (stats.norm.cdf(X, mu, 1)) * 2
y = (1 - stats.norm.cdf(X, 0, 1)) * 2

plt.plot(X, y)
# plt.xlim(x_min, x_max)
# plt.xlabel('wallet guess')
plt.xlabel('number of nodes with purchasing ability guess')
plt.ylabel('probability')
# plt.title('pdf of normal distribution: μ = mean of product prices, σ = μ / 4')
# # plt.title('cdf of normal distribution: μ = mean of product prices, σ = μ / 4')
# # plt.title('ccdf of normal distribution: μ = mean of product prices, σ = μ / 4')
# plt.title('pdf of normal distribution: μ = 0, σ = 1')
# plt.title('cdf of normal distribution: μ = 0, σ = 1')
plt.title('ccdf of normal distribution: μ = 0, σ = 1')
plt.grid()
plt.show()

print(y)
pw_list = [round(float(y[np.argwhere(X == p)]), 4) for p in price_list[pk]]
print(pw_list)