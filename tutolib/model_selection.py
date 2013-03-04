from collections import namedtuple
from collections import defaultdict

from IPython.parallel import interactive
from scipy.stats import sem
import numpy as np
from sklearn.grid_search import IterGrid
from sklearn.utils import check_random_state
from sklearn.cross_validation import ShuffleSplit

from tutolib.mmap import warm_mmap_on_cv_splits


@interactive
def compute_evaluation(model, cv_split_filename, params=None,
    train_fraction=1.0, mmap_mode='r'):
    """Function executed on a worker to evaluate a model on a given CV split"""
    # All module imports should be executed in the worker namespace
    from time import time
    from sklearn.externals import joblib
    
    X_train, y_train, X_test, y_test = joblib.load(
        cv_split_filename, mmap_mode=mmap_mode)

    # Slice a subset of the training set for plotting learning curves 
    n_samples_train = int(train_fraction * X_train.shape[0])
    X_train = X_train[:n_samples_train]
    y_train = y_train[:n_samples_train]

    # Configure the model
    if model is not None:
        model.set_params(**params)

    # Fit model and measure training time
    t0 = time()
    model.fit(X_train, y_train)
    train_time = time() - t0
    
    # Compute score on training set
    train_score = model.score(X_train, y_train)
    
    # Compute score on test set
    test_score = model.score(X_test, y_test)

    # Wrap evaluation results in a simple tuple datastructure
    return (test_score, train_score, train_time,
            train_fraction, params)


# Named tuple to collect evaluation results
Evaluation = namedtuple('Evaluation', (
    'validation_score',
    'train_score',
    'train_time',
    'train_fraction',
    'parameters'))


class RandomizedGridSeach(object):
    """"Async Randomized Parameter search."""

    def __init__(self, load_balanced_view, random_state=0):
        self.task_groups = []
        self.lb_view = load_balanced_view
        self.random_state = random_state

    def map_tasks(self, f):
        return [f(task) for task_group in self.task_groups
                        for task in task_group]

    def abort(self):
        for task_group in self.task_groups:
            for task in task_group:
                if not task.ready():
                    try:
                        task.abort()
                    except AssertionError:
                        pass
        return self

    def wait(self):
        self.map_tasks(lambda t: t.wait())
        return self

    def completed(self):
        return sum(self.map_tasks(lambda t: t.ready()))

    def total(self):
        return sum(self.map_tasks(lambda t: 1))

    def progress(self):
        c = self.completed()
        if c == 0:
            return 0.0
        else:
            return float(c) / self.total()

    def reset(self):
        # Abort any other previously scheduled tasks
        self.map_tasks(lambda t: t.abort())

        # Schedule a new batch of evalutation tasks
        self.task_groups, self.all_parameters = [], []

    def launch_for_splits(self, model, parameter_grid, cv_split_filenames,
        pre_warm=True):
        """Launch a Grid Search on precomputed CV splits."""

        # Abort any existing processing and erase previous state
        self.reset()

        # Warm the OS disk cache on each host with sequential reads
        # XXX: fix me: interactive namespace issues to resolve
        # if pre_warm:
        #     warm_mmap_on_cv_splits(self.lb_view.client, cv_split_filenames)

        # Randomize the grid order
        random_state = check_random_state(self.random_state)
        self.all_parameters = list(IterGrid(parameter_grid))
        random_state.shuffle(self.all_parameters)

        for params in self.all_parameters:
            task_group = []
            
            for cv_split_filename in cv_split_filenames:
                task = self.lb_view.apply(compute_evaluation,
                    model, cv_split_filename, params=params)
                task_group.append(task)

            self.task_groups.append(task_group)

        # Make it possible to chain method calls
        return self

    def find_bests(self, n_top=5):
        """Compute the mean score of the completed tasks"""
        mean_scores = []
        
        for params, task_group in zip(self.all_parameters, self.task_groups):
            scores = [Evaluation(*t.get()).validation_score
                      for t in task_group if t.ready()]
            if len(scores) == 0:
                continue
            mean_scores.append((np.mean(scores), sem(scores), params))
                       
        return sorted(mean_scores, reverse=True)[:n_top]

    def report(self, n_top=5):
        bests = self.find_bests()
        output = "Progress: {0}% ({1:03d}/{2:03d})\n".format(
            self.progress(), self.completed(), self.total())
        for i, best in enumerate(bests):
            output += "\nRank {0}: {1:.5f} (+/-{2:.5f}): {3}".format(
                i + 1, *best)
        return output

    def __repr__(self):
        return self.report()
