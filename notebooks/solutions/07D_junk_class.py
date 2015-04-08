# Generate random text document by mixing lines irrespective
# of the class
import random
random.seed(0)

all_lines = [line for doc in all_twenty_train.data
                  for line in doc.split('\n')]

n_junk_docs = 1000
n_lines_per_doc = 1000

junk_docs = []
for i in range(n_junk_docs):
    junk_doc = "\n".join(random.sample(all_lines, n_lines_per_doc))
    junk_docs.append(junk_doc)

#Concatenate the new junk documentation to the previous dataset
new_data_train = twenty_train_small.data + junk_docs

junk_targets = [4] * len(junk_docs)
new_target_train = np.concatenate([twenty_train_small.target,
                                   junk_targets])

new_target_names = np.concatenate([twenty_train_small.target_names,
                                   ['junk']])


# Retrain the pipeline 
_ = pipeline.fit(new_data_train, new_target_train)

vec_name, vec = pipeline.steps[0]
clf_name, clf = pipeline.steps[1]

feature_names = vec.get_feature_names()
feature_weights = clf.coef_
        
print(display_important_features(feature_names, new_target_names,
                                 feature_weights))
