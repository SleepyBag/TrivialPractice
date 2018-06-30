from scipy import stats
import matplotlib.pyplot as plt
from math import *


def regression(x, y, x0=0):
    plt.plot(x, y, '.')
    plt.show()
    n = len(x)
    sxx, syy, sxy = (0, 0, 0)
    sumx, sumy = (0, 0)
    for i in x:
        sxx += i**2
        sumx += i
    for i in y:
        syy += i**2
        sumy += i
    for i in range(len(x)):
        sxy += x[i]*y[i]
    sxx -= sumx**2/n
    syy -= sumy**2/n
    sxy -= sumx*sumy/n

    bhat = sxy/sxx
    ahat = sumy/n-(sumx/n)*bhat
    print('y={ahat:.2e}+{bhat:.2e}x'.format(ahat=ahat, bhat=bhat))

    sigma2hat = (syy-bhat*sxy)/(n-2)
    sigmahat = sqrt(sigma2hat)
    print('sigma方的无偏估计为', sigma2hat)

    t = abs(bhat)/sqrt(sigma2hat)*sqrt(sxx)
    t_dis = stats.t(df=n-2)
    prob = (1-t_dis.cdf(t))*2
    print('接受H0的最高显著性为', prob)

    deltab = t_dis.ppf(.975)*sqrt(sigma2hat)/sqrt(sxx)
    print('b的置信水平为0.95的置信区间为({0:.2f},{1:.2f})'.format(
        bhat-deltab, bhat+deltab))

    deltamiu = t_dis.ppf(.975)*sigmahat*sqrt(1/n+(x0-sumx/n)**2/sxx)
    print('miu({x0})的置信水平为0.95的置信区间为({0:.2f},{1:.2f})'.format(
        (ahat+bhat*x0)-deltamiu, (ahat+bhat*x0)+deltamiu,x0=x0))

    deltay = t_dis.ppf(.975)*sigmahat*sqrt(1+1/n+(x0-sumx/n)**2/sxx)
    print('Y({x0})的置信水平为0.95的置信区间为({0:.2f},{1:.2f})'.format(
        (ahat+bhat*x0)-deltay, (ahat+bhat*x0)+deltay,x0=x0))
