import numpy as np
from matplotlib import pyplot as plt


def do_r2(results, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    r = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    y_pos = np.arange(len(r))
    ax.barh(y_pos, width=r.values(), height=.6, align='center', color=['#5170d7'], alpha=.8)
    ax.set_yticks(y_pos, labels=r.keys())
    ax.set_xlim((0, 1))
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.invert_yaxis()
    ax.set_xlabel('R²')
    ax.set_title('SMART16 | PM₁₀ | Regressors performance (R²)')
    for i, v in enumerate(r.values()):
        ax.text(v, i, " " + str(v), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('../thesis_img/hist_' + filename, dpi=300)


def do_rmse(results, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    r = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=False)}
    y_pos = np.arange(len(r))
    ax.barh(y_pos, width=r.values(), height=.6, align='center', color=['#fdaa48'], alpha=.8)
    ax.set_yticks(y_pos, labels=r.keys())
    ax.set_xlim((0, 1))
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    ax.invert_yaxis()
    ax.set_xlabel('µg/m³')
    ax.set_title('SMART16 | PM₁₀ | Regressors performance (RMSE)')
    for i, v in enumerate(r.values()):
        ax.text(v, i, " " + str(v), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('../thesis_img/hist_' + filename, dpi=300)


if __name__ == '__main__':
    results = {
        'Lineare': 0.715,
        'Lineare robusto (Huber)': 0.716,
        'Lineare avanzato (Cook)': 0.785,
        'Ridge': 0.715,
        'Lasso': 0.716,
        'Polinomiale grado 2': 0.739,
        'Polinomiale grado 3': 0.738,
        'Random Forest': 0.590,
        'Gradient Boosting': 0.238,
        'SVR (Kernel lineare)': 0.707,
        'SVR (Kernel polinomiale)': 0.513,
        'SVR (Kernel RBF)': 0.719,
        'KernelRidge': 0.738
    }
    do_r2(results, '1')

    results = {
        'Lineare': 9.706,
        'Lineare robusto (Huber)': 9.760,
        'Lineare avanzato (Cook)': 7.103,
        'Ridge': 9.707,
        'Lasso': 9.754,
        'Polinomiale grado 2': 9.343,
        'Polinomiale grado 3': 9.339,
        'Random Forest': 11.686,
        'Gradient Boosting': 16.004,
        'SVR (Kernel lineare)': 9.860,
        'SVR (Kernel polinomiale)': 12.724,
        'SVR (Kernel RBF)': 9.672,
        'KernelRidge': 9.321
    }
    do_rmse(results, '2')
