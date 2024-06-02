from train_model import X_train, X_test, y_train, y_test
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error


param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5]
}


catboost_model = CatBoostRegressor(verbose=0, random_seed=42)


grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=3, n_jobs=-1)


grid_search.fit(X_train, y_train)

# Лучшая модель и параметры
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'Лучшие параметры: {best_params}')


y_pred = best_model.predict(X_test)


mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
