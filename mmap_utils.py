import os
from IPython.parallel import interactive


@interactive
def persist_cv_splits(X, y, name=None, n_cv_iter=5, suffix="_cv_%03d.pkl",
                      train_size=None, test_size=0.25, random_state=None,
                      folder='.'):
    """Materialize randomized train test splits of a dataset."""
    from sklearn.externals import joblib
    from sklearn.cross_validation import ShuffleSplit
    import os
    import uuid

    if name is None:
        u = uuid.uuid4()
        if hasattr(u, 'get_hex'):
            # Python 2 compat
            name = u.get_hex()
        else:
            name = u.hex

    cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
        test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    for i, (train, test) in enumerate(cv):
        cv_fold = (X[train], y[train], X[test], y[test])
        cv_split_filename = os.path.join(folder, name + suffix % i)
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(cv_fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)

    return cv_split_filenames


def warm_mmap_on_cv_splits(client, cv_split_filenames):
    """Trigger a disk load on all the arrays of the CV splits

    Assume the files are shared on all the hosts using NFS.
    """
    # First step: query cluster to fetch one engine id per host
    all_engines = client[:]

    @interactive
    def hostname():
        import socket
        return socket.gethostname()

    hostnames = all_engines.apply(hostname).get_dict()
    one_engine_per_host = dict((hostname, engine_id)
                               for engine_id, hostname
                               in hostnames.items())
    one_engine_per_host_ids = list(one_engine_per_host.values())
    hosts_view = client[one_engine_per_host_ids]

    # Second step: for each data file and host, mmap the arrays of the file
    # and trigger a sequential read of all the arrays' data
    @interactive
    def load_in_memory(filenames):
        from sklearn.externals import joblib
        for filename in filenames:
            arrays = joblib.load(filename, mmap_mode='r')
            for array in arrays:
                array.sum()  # trigger the disk read

    cv_split_filenames = [os.path.abspath(f) for f in cv_split_filenames]
    hosts_view.apply_sync(load_in_memory, cv_split_filenames)
