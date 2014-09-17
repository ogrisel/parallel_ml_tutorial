strip_vectorizer = TfidfVectorizer(preprocessor=strip_headers, min_df=2)
X_train_small_stripped = strip_vectorizer.fit_transform(
    twenty_train_small.data)

y_train_small_stripped = twenty_train_small.target

classifier = MultinomialNB().fit(
  X_train_small_stripped, y_train_small_stripped)

print("Training score: {0:.1f}%".format(
    classifier.score(X_train_small_stripped, y_train_small_stripped) * 100))

X_test_small_stripped = strip_vectorizer.transform(twenty_test_small.data)
y_test_small_stripped = twenty_test_small.target
print("Testing score: {0:.1f}%".format(
    classifier.score(X_test_small_stripped, y_test_small_stripped) * 100))

# Analysis:
# So indeed the header data is making the problem easier (cheating one could
# say) but naive Bayes classifier can still guess 80% of the time against
# 1 / 4 == 25% mean score for a random guessing on the small subset with
# 4 target categories.
