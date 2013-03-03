from collections import namedtuple
from collections import defaultdict
from scipy.stats import sem
from sklearn.cross_validation import ShuffleSplit


def compute_evaluation(model, cv_split_filename, params=None):
    """Function executed on a worker to evaluate a model on a given CV split"""
    # All module imports should be executed in the worker namespace
    from time import time
    from sklearn.externals import joblib
    
    X_train, y_train, X_test, y_test = joblib.load(
        cv_split_filename, mmap_mode='c')

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
            X_train.shape[0], cv_index, params)


# Named tuple to collect evaluation results
Evaluation = namedtuple('Evaluation', (
    'test_score',
    'train_score',
    'train_time',
    'train_size',
    'cv_index',
    'parameters'))


class LearningCurves(object):
    """Handle async, distributed evaluation of a model learning curves"""
    
    def __init__(self, load_balancer):
        self._scheduled_tasks = []
        self._load_balancer = load_balancer

    def evaluate(self, model, X, y, train_sizes=np.linspace(0.1, 0.8, 8),
        test_size=0.2, n_iter=5):
        """Schedule evaluation work in parallel and aggregate results for plotting"""
        
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        
        # Abort any other previously scheduled tasks
        for task in self._scheduled_tasks:
            if not task.ready():
                task.abort()

        # Schedule a new batch of evalutation tasks
        self._scheduled_tasks = []
        for train_size in train_sizes:
            cv = ShuffleSplit(n_samples, n_iter=n_iter, 
                              train_size=train_size,
                              test_size=test_size)
            for cv_index, (train, test) in enumerate(cv):
                task = self._load_balancer.apply_async(
                    compute_evaluation,
                    clf, X[train], y[train], X[test], y[test],
                    cv_index)
                
                self._scheduled_tasks.append(task)
        
        # Make it possible to chain method calls
        return self
                
    def wait(self):
        """Wait for completion"""
        for task in self._scheduled_tasks:
            task.wait()
        
        # Make it possible to chain method calls
        return self
    
    def update_summary(self):
        """Compute summary statistics for all finished tasks"""
        evaluations = [Evaluation(*t.get())
                       for t in self._scheduled_tasks if t.ready()]
        grouped_evaluations = defaultdict(list)
        for ev in evaluations:
            # Group evaluations by effective training sizes
            grouped_evaluations[ev.train_size].append(ev)
        
        self.train_sizes = []
        self.train_scores, self.train_scores_stderr = [], []
        self.test_scores, self.test_scores_stderr = [], []
        self.train_times, self.train_times_stderr = [], []
    
        for size, group in sorted(grouped_evaluations.items()):
            self.train_sizes.append(size)
    
            # Aggregate training scores data
            train_scores = [ev.train_score for ev in group]
            self.train_scores.append(np.mean(train_scores))
            self.train_scores_stderr.append(sem(train_scores))
    
            # Aggregate testing scores data
            test_scores = [ev.test_score for ev in group]
            self.test_scores.append(np.mean(test_scores))
            self.test_scores_stderr.append(sem(test_scores))
            
            # Aggregate training times data
            train_times = [ev.train_time for ev in group]
            self.train_times.append(np.mean(train_times))
            self.train_times_stderr.append(sem(train_times))
            
        self.train_sizes = np.asarray(self.train_sizes)
        self.train_scores = np.asarray(self.train_scores)
        self.train_scores_stderr = np.asarray(self.train_scores_stderr)
        self.test_scores = np.asarray(self.test_scores)
        self.test_scores_stderr = np.asarray(self.test_scores_stderr)
        self.train_times = np.asarray(self.train_times)
        self.train_times_stderr = np.asarray(self.train_times_stderr)
            
    def __repr__(self):
        """Display current evaluation statistics"""
        self.update_summary()
        if self.test_scores.shape[0] == 0:
            return "Missing evaluation statistics"
        n_total = len(self._scheduled_tasks)
        n_done = len([t for t in self._scheduled_tasks if t.ready()])
        test, test_stderr = self.test_scores[-1], self.test_scores_stderr[-1]
        train, train_stderr = self.train_scores[-1], self.train_scores_stderr[-1]
        time, time_stderr = self.train_times[-1], self.train_times_stderr[-1]
        overfitting = np.max(train - test, 0)
        underfitting = np.max(1 - train, 0)
        return (
                "Progress: {n_done}/{n_total} CV tasks\n"
                "Last train score: {train:.5f} (+/-{train_stderr:.5f})\n"
                "Last test score: {test:.5f} (+/-{test_stderr:.5f})\n"
                "Last train time: {time:.3f}s (+/-{time_stderr:.3f})\n"
                "Overfitting: {overfitting:.5f}\n"
                "Underfitting: {underfitting:.5f}\n"
        ).format(**locals())


#pl.fill_between(train_sizes, mean_test - confidence, mean_test + confidence,
#                color = 'b', alpha = .2)
#pl.plot(train_sizes, mean_test, 'o-k', c='b', label = 'Test score')

def plot_learning_curves(lc):
    """Interative plot of a learning curve"""
    lc.update_summary()  # ensure that stats are up to date
    pl.figure()
    if hasattr(lc, 'train_times'):
        pl.subplot(211)
    
    pl.fill_between(lc.train_sizes,
                    lc.train_scores - 2 * lc.train_scores_stderr,
                    lc.train_scores + 2 * lc.train_scores_stderr,
                    color = 'g', alpha = .2)
    pl.plot(lc.train_sizes, lc.train_scores, 'o-k', c='g',
            label = 'Train score')
    
    pl.fill_between(lc.train_sizes,
                    lc.test_scores - 2 * lc.test_scores_stderr,
                    lc.test_scores + 2 * lc.test_scores_stderr,
                    color = 'b', alpha = .2)
    pl.plot(lc.train_sizes, lc.test_scores, 'o-k', c='b',
            label = 'Test score')
    
    pl.ylabel('Score')
    pl.xlim(0, lc.n_samples)
    pl.ylim((None, 1.01))  # The best possible score is 1.0
    pl.legend(loc='best')
    pl.title('Main train and test scores +/- 2 standard errors')
    
    if hasattr(lc, 'train_times'):
        pl.subplot(212)
        pl.errorbar(lc.train_sizes,
                    lc.train_times,
                    np.asarray(lc.train_times_stderr) * 2)
        pl.xlim(0, lc.n_samples)
        pl.ylabel('Training time (s)')
        pl.xlabel('# training samples')
