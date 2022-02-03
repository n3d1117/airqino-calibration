import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6, 3.5))
    results = {
        'Linear': 0.599,
        'Linear Robust (Cook)': 0.660,
        'Linear Robust (Huber)': 0.597,
        'Polynomial': 0.654,
        'Random Forest': 0.464,
        'Gradient Boosting': 0.213,
        'SVR (Linear Kernel)': 0.580,
        'SVR (Polynomial Kernel)': 0.321,
        'SVR (RBF Kernel)': 0.627
    }
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    y_pos = np.arange(len(results))
    ax.barh(y_pos, width=results.values(), height=.5, align='center', color=['tab:blue'], alpha=.8)
    ax.set_yticks(y_pos, labels=results.keys())
    ax.set_xlim((0, 1))
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.invert_yaxis()
    ax.set_xlabel('R²')
    ax.set_title('SMART16 | PM10 | Regressors performance')
    for i, v in enumerate(results.values()):
        ax.text(v, i, " " + str(v), va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
