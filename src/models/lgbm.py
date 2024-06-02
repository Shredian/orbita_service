from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from train_model import X_train, X_test, y_train, y_test

param_grid = {
    'num_leaves': [30, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500]
}

lgbm_model = LGBMRegressor(random_state=42)


grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=3, n_jobs=-1)


grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'Лучшие параметры: {best_params}')


y_pred = best_model.predict(X_test)


mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
