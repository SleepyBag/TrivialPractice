from scipy import stats
from math import sqrt


def analyze(statistics, alpha=0, belief=0):

    r, s, t = (len(statistics), len(statistics[0]), len(statistics[0][0]))
    T = 0
    Tij = [[0 for i in range(s)] for i in range(r)]
    Ti = [0] * r
    Tj = [0] * s

    for (i, sample_a) in enumerate(statistics):
        for (j, sample_b) in enumerate(sample_a):
            for (k, num) in enumerate(sample_b):
                T += num
                Tij[i][j] += num
                Ti[i] += num
                Tj[j] += num

    st = 0
    for i in range(r):
        for j in range(s):
            for k in range(t):
                st += statistics[i][j][k] ** 2
    st -= T ** 2 / r / s / t

    sa = 0
    for i in range(r):
        sa += Ti[i] ** 2
    sa /= s * t
    sa -= T ** 2 / r / s / t

    sb = 0
    for j in range(s):
        sb += Tj[j] ** 2
    sb /= r * t
    sb -= T ** 2 / r / s / t

    sapb = 0
    if t != 1:
        for i in range(r):
            for j in range(s):
                sapb += Tij[i][j] ** 2
        sapb /= t
        sapb -= T ** 2 / r / s / t
        sapb -= sa + sb

    se = st - sa - sb - sapb

    sab = sa / (r - 1)
    sbb = sb / (s - 1)
    sapbb = sapb / (r - 1) / (s - 1)
    if(t!=1):
        seb = se / r / s / (t - 1)
    else:
        seb = se / (r - 1) / (s - 1)
    fa = sab / seb
    fb = sbb / seb
    fapb = sapbb / seb

    # f_distribution = stats.f(s-1, n-s)

    print('方差来源', '平方和', '\t自由度', '均方', 'F比', sep='\t')
    print('因素A', '\t\t{0:.2f}\t\t{1:.2f}\t{2:.2f}\t{3:.2f}\t'.format(
        sa, r-1, sab, fa))
    print('因素B', '\t\t{0:.2f}\t\t{1:.2f}\t{2:.2f}\t{3:.2f}\t'.format(
        sb, s-1, sbb, fb))
    if(t!=0):
        print('交互作用', '\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t'.format(
            sapb, (r-1)*(s-1), sapbb, fapb))
        print('误差', '\t\t{0:.2f}\t{1:.2f}\t{2:.2f}\t'.format(se, r*s*(t-1), seb))
    print('总和', '\t\t{0:.2f}\t\t{1:.2f}\t'.format(st, r*s*t-1))

    # if f >= f_distribution.ppf(alpha):
    # print('有显著差异')
    # else:
    # print('没有显著差异')
    # for j in range(len(statistics)):
    # for k in range(j+1, len(statistics)):
    # if j != k:
    # t = stats.t(n-s)
    # delta = t.ppf(1-belief) * sqrt(seb * (1/len(statistics[j]) + 1/len(statistics[k])))
    # print('miu%d-miu%d置信区间:' % (j, k),
    # '(%.2f,%2f)' % (xbaj[j]-xbaj[k]-delta, xbaj[j]-xbaj[k]+delta))
    # return t
