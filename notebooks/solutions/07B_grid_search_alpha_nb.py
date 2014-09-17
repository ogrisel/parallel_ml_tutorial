from sklearn.grid_search import GridSearchCV
from pprint import pprint

nb_params = {
    'alpha': np.logspace(-3, 3, 7),
}

gs = GridSearchCV(MultinomialNB(), nb_params, cv=5, n_jobs=-1)
gs.fit(X_train_small_stripped, y_train_small_stripped)

pprint(gs.grid_scores_)
print('Best alpha: %r' % gs.best_params_['alpha'])
