import numpy as np
import os
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen
import tarfile
import zipfile
import gzip
from sklearn.datasets import load_files
from sklearn.externals import joblib


TWENTY_URL = ("http://people.csail.mit.edu/jrennie/"
              "20Newsgroups/20news-bydate.tar.gz")
TWENTY_ARCHIVE_NAME = "20news-bydate.tar.gz"
TWENTY_CACHE_NAME = "20news-bydate.pkz"
TWENTY_TRAIN_FOLDER = "20news-bydate-train"
TWENTY_TEST_FOLDER = "20news-bydate-test"

SENTIMENT140_URL = ("http://cs.stanford.edu/people/alecmgo/"
                    "trainingandtestdata.zip")
SENTIMENT140_ARCHIVE_NAME = "trainingandtestdata.zip"


COVERTYPE_URL = ('http://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/covtype/covtype.data.gz')

# Source: https://www.kaggle.com/c/titanic-gettingStarted/data
TITANIC_URL = ("https://dl.dropboxusercontent.com/"
               "u/5743203/data/titanic/titanic_train.csv")


def get_datasets_folder():
    here = os.path.dirname(__file__)
    datasets_folder = os.path.abspath(os.path.join(here, 'datasets'))
    datasets_archive = os.path.abspath(os.path.join(here, 'datasets.zip'))

    if not os.path.exists(datasets_folder):
        if os.path.exists(datasets_archive):
            print("Extracting " + datasets_archive)
            zf = zipfile.ZipFile(datasets_archive)
            zf.extractall('.')
            assert os.path.exists(datasets_folder)
        else:
            print("Creating datasets folder: " + datasets_folder)
            os.makedirs(datasets_folder)
    else:
        print("Using existing dataset folder:" + datasets_folder)
    return datasets_folder


def check_twenty_newsgroups(datasets_folder):
    print("Checking availability of the 20 newsgroups dataset")

    archive_path = os.path.join(datasets_folder, TWENTY_ARCHIVE_NAME)
    train_path = os.path.join(datasets_folder, TWENTY_TRAIN_FOLDER)
    test_path = os.path.join(datasets_folder, TWENTY_TEST_FOLDER)


    if not os.path.exists(archive_path):
        print("Downloading dataset from %s (14 MB)" % TWENTY_URL)
        opener = urlopen(TWENTY_URL)
        open(archive_path, 'wb').write(opener.read())
    else:
        print("Found archive: " + archive_path)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Decompressing %s" % archive_path)
        tarfile.open(archive_path, "r:gz").extractall(path=datasets_folder)

    print("Checking that the 20 newsgroups files exist...")
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    print("=> Success!")


def check_sentiment140(datasets_folder):
    print("Checking availability of the sentiment 140 dataset")
    archive_path = os.path.join(datasets_folder, SENTIMENT140_ARCHIVE_NAME)
    sentiment140_path = os.path.join(datasets_folder, 'sentiment140')
    train_path = os.path.join(sentiment140_path,
        'training.1600000.processed.noemoticon.csv')
    test_path = os.path.join(sentiment140_path,
        'testdata.manual.2009.06.14.csv')

    if not os.path.exists(archive_path):
        print("Downloading dataset from %s (77MB)" % SENTIMENT140_URL)
        opener = urlopen(SENTIMENT140_URL)
        open(archive_path, 'wb').write(opener.read())
    else:
        print("Found archive: " + archive_path)

    if not os.path.exists(sentiment140_path):
        print("Extracting %s to %s" % (archive_path, sentiment140_path))
        zf = zipfile.ZipFile(archive_path)
        zf.extractall(sentiment140_path)
    print("Checking that the sentiment 140 CSV files exist...")
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    print("=> Success!")


def check_covertype(datasets_folder):
    print("Checking availability of the covertype dataset")
    archive_path = os.path.join(datasets_folder, 'covtype.data.gz')
    covtype_dir = os.path.join(datasets_folder, "covertype")
    samples_path = os.path.join(covtype_dir, "samples.pkl")
    targets_path = os.path.join(covtype_dir, "targets.pkl")

    if not os.path.exists(covtype_dir):
        os.makedirs(covtype_dir)

    if not os.path.exists(archive_path):
        print("Downloading dataset from %s (10.7MB)" % COVERTYPE_URL)
        open(archive_path, 'wb').write(urlopen(COVERTYPE_URL).read())
    else:
        print("Found archive: " + archive_path)

    if not os.path.exists(samples_path) or not os.path.exists(targets_path):
        print("Parsing the data and splitting input and labels...")
        f = open(archive_path, 'rb')
        Xy = np.genfromtxt(gzip.GzipFile(fileobj=f), delimiter=',')

        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32)

        joblib.dump(X, samples_path)
        joblib.dump(y, targets_path )
    print("=> Success!")


def check_titanic(datasets_folder):
    print("Checking availability of the titanic dataset")
    csv_filename = os.path.join(datasets_folder, 'titanic_train.csv')
    if not os.path.exists(csv_filename):
        print("Downloading titanic data from %s" % TITANIC_URL)
        open(csv_filename, 'wb').write(urlopen(TITANIC_URL).read())
    print("=> Success!")


if __name__ == "__main__":
    import sys
    datasets_folder = get_datasets_folder()
    check_twenty_newsgroups(datasets_folder)
    check_titanic(datasets_folder)
    if 'sentiment140' in sys.argv:
        check_sentiment140(datasets_folder)
    if 'covertype' in sys.argv:
        check_covertype(datasets_folder)