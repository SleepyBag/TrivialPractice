from scipy import stats
from math import sqrt


def analyze(statistics, alpha=0, belief=0):
    belief = 1 - belief
    belief /= 2
    taj = []
    xbaj = []
    taa = 0
    st = 0
    sa = 0
    n = 0
    s = len(statistics)
    for sample in statistics:
        n += len(sample)
        tmptaj = 0
        for num in sample:
            tmptaj += num
            taa += num
            st += num ** 2
        taj.append(tmptaj)
        xbaj.append(tmptaj / len(sample))
        sa += tmptaj ** 2 / len(sample)
    st -= taa ** 2 / n
    sa -= taa ** 2 / n
    se = st - sa
    sab = sa / (s - 1)
    seb = se / (n - s)
    f = sab / seb
    f_distribution = stats.f(s-1, n-s)
    print('方差来源', '平方和', '自由度', '均方', 'F比', sep='\t')
    print('因素A', '\t%.2f\t%.2f\t%.2f\t%.2f\t' % (sa, s-1, sab, f), sep='\t')
    print('误差', '\t%.2f\t%.2f\t%.2f\t' % (se, n-s, seb), sep='\t')
    print('总和', '\t%.2f\t%.2f\t' % (st, n-1), sep='\t')
    if f >= f_distribution.ppf(alpha):
        print('有显著差异')
    else:
        print('没有显著差异')
    for j in range(len(statistics)):
        for k in range(j+1, len(statistics)):
            if j != k:
                t = stats.t(n-s)
                delta = t.ppf(1-belief) * sqrt(seb * (1/len(statistics[j]) + 1/len(statistics[k])))
                print('miu%d-miu%d置信区间:' % (j, k),
                      '(%.2f,%2f)' % (xbaj[j]-xbaj[k]-delta, xbaj[j]-xbaj[k]+delta))
    return t
