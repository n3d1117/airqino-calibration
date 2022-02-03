from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from regression.regression_models import get_polynomial_model
from utils.utils import get_dataset, Dataset


def avg(scores):
    return '%0.4f' % (sum(scores) / len(scores))


# y = a + b*x + c*x^2
def make_avg(predictor, X, y, n=5000):
    r2_scores, rmse_scores, a, b, c = [], [], [], [], []
    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        predictor.fit(X_train, y_train)
        # print(predictor.steps[0][1].get_feature_names_out())
        a.append(predictor.steps[1][1].intercept_)
        b.append(predictor.steps[1][1].coef_[0])
        c.append(predictor.steps[1][1].coef_[1])
        r2_scores.append(predictor.score(X_test, y_test))
        rmse_scores.append(mean_squared_error(y_test, predictor.predict(X_test), squared=False))
    return avg(r2_scores), avg(rmse_scores), avg(a), avg(b), avg(c)


def apply_poly_regression(X, y):
    return make_avg(get_polynomial_model(), X, y)


def print_annual_results(results, title):
    table = PrettyTable()
    table.title = title
    table.add_column('Regressor', ['Polynomial'])
    table.add_column('R²', results[0])
    table.add_column('RMSE', results[1])
    table.add_column('a', results[2])
    table.add_column('b', results[3])
    table.add_column('c', results[4])
    table.align = 'l'
    # table.set_style(MARKDOWN)
    print(table)


def evaluate(X, y):
    poly_score, poly_rmse, a, b, c = apply_poly_regression(X, y)
    r2_scores = [poly_score]
    rmse_scores = [poly_rmse]
    return r2_scores, rmse_scores, [a], [b], [c]


def annual_summary(X, y, station, chemical):
    results = evaluate(X, y)
    print_annual_results(results, title='All year | {n} | {c}'.format(c=chemical.upper(), n=station))


if __name__ == '__main__':
    dataset = get_dataset(Dataset.SMART16_NEW_PM_8H)
    X = dataset['airqino_{}'.format('pm2.5')].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format('pm2.5')].values
    annual_summary(X, y, 'SMART16_new-CAPANNORI-8h', 'pm2.5')
