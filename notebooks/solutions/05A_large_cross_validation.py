cv = ShuffleSplit(n_samples, n_iter=50, train_size=500, test_size=500,
    random_state=0)
%time scores = cross_val_score(SVC(C=10, gamma=0.005), X, y, cv=cv)
print(mean_score(scores))
