gb_new = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    subsample=.8, max_features=.5)
gb_new.fit(features, target)
feature_names = features.columns.values
x = np.arange(len(feature_names))
plt.bar(x, gb_new.feature_importances_)
_ = plt.xticks(x + 0.5, feature_names, rotation=30)
