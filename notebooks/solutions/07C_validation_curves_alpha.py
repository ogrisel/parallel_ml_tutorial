from sklearn.learning_curve import validation_curve

alpha_range = np.logspace(-6, 0, 7)

train_scores, validation_scores = validation_curve(
    MultinomialNB(), X_train_small_stripped, y_train_small_stripped,
    'alpha', alpha_range, cv=5, n_jobs=-1)

plt.semilogx(alpha_range, train_scores.mean(axis=1), label='train')
plt.semilogx(alpha_range, validation_scores.mean(axis=1), label='validation')
plt.legend(loc='best')
_ = plt.title('Validation curves')

# Analysis:
# For low values of alpha (no smoothing), the model is not biased and hence free
# to overfit. Smoothing a bit with `alpha=0.001` or `alpha=0.01` makes the
# validation score increase a bit (thus overfitting a bit less but not by much).
# If alpha is too strong the model is too biased or constrained and underfits.
