import numpy as np

def CohenEffectSize(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: statistic and standard error
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)

    std_d = np.sqrt(((n1 + n2)/(n1 * n2) + (d**2)/(2*(n1 + n2 - 2)))*((n1 + n2)/(n1 + n2 -2)))

    return d, std_d
