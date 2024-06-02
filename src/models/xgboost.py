from train_model import X_train, X_test, y_train, y_test
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}


xgb_model = XGBRegressor(random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'Лучшие параметры: {best_params}')

y_pred = best_model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')

model_path = 'models/xgb_model.json'
best_model.save_model(model_path)
print(f'Модель сохранена по пути: {model_path}')
