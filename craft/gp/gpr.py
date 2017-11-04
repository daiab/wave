# ref: https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
# good doc: https://github.com/thuijskens/bayesian-optimization
# good book: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
# paper: https://arxiv.org/abs/1610.08733

import numpy as np
import matplotlib.pylab as plt

COV_PARAM = [1, 10]


def exp_cov(x, y):
    return COV_PARAM[0] * np.exp(-0.5 * COV_PARAM[1] * np.subtract.outer(x, y) ** 2)


def conditional(x_new, x, y):
    A = exp_cov(x_new, x_new)
    B = exp_cov(x_new, x)
    C = exp_cov(x, x)
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return mu.squeeze(), sigma.squeeze()


sigma_0 = exp_cov(0, 0)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)
plt.show()

x_point = [1.]
y_point = [np.random.normal(scale=sigma_0)]
print(y_point)

x_point_sigma = exp_cov(x_point, x_point)


def predict(x, ori_x, ori_sigma, ori_y):
    new_sigma = [exp_cov(x, ox) for ox in ori_x]
    ori_sigma_inv = np.linalg.inv(ori_sigma)
    y_pred = np.dot(new_sigma, ori_sigma_inv).dot(ori_y)
    sigma_new = exp_cov(x, x) - np.dot(new_sigma, ori_sigma_inv).dot(new_sigma)
    return y_pred, sigma_new


x_linspace = np.linspace(-3, 3, 1000)
predictions = [predict(xr, x_point, x_point_sigma, y_point) for xr in x_linspace]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_linspace, y_pred, yerr=sigmas, capsize=0)
plt.plot(x_point, y_point, "ro")
plt.show()

m, s = conditional([-0.7], x_point, y_point)
y_2 = np.random.normal(m, s)
print(y_2)

x_point.append(-0.7)
y_point.append(y_2)

x_point_sigma = exp_cov(x_point, x_point)
predictions = [predict(xr, x_point, x_point_sigma, y_point) for xr in x_linspace]
y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_linspace, y_pred, yerr=sigmas, capsize=0)
plt.plot(x_point, y_point, "ro")
plt.show()

x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
mu, s = conditional(x_more, x_point, x_point)
y_more = np.random.multivariate_normal(mu, s)
print(y_more)

x_point += x_more
y_point += y_more.tolist()

x_point_sigma = exp_cov(x_point, x_point)
predictions = [predict(i, x_point, x_point_sigma, y_point) for i in x_linspace]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_linspace, y_pred, yerr=sigmas, capsize=0)
plt.plot(x_point, y_point, "ro")
plt.show()
