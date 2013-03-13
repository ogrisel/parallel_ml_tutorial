import os
import urllib
import tarfile
from sklearn.datasets import load_files

TWENTY_URL = ("http://people.csail.mit.edu/jrennie/"
              "20Newsgroups/20news-bydate.tar.gz")
TWENTY_ARCHIVE_NAME = "20news-bydate.tar.gz"
TWENTY_CACHE_NAME = "20news-bydate.pkz"
TWENTY_TRAIN_FOLDER = "20news-bydate-train"
TWENTY_TEST_FOLDER = "20news-bydate-test"


def get_datasets_folder():
    here = os.path.dirname(__file__)
    datasets_folder = os.path.abspath(os.path.join(here, 'datasets'))

    if not os.path.exists(datasets_folder):
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
        opener = urllib.urlopen(TWENTY_URL)
        open(archive_path, 'wb').write(opener.read())
    else:
        print("Found archive: " + archive_path)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Decompressing %s" % archive_path)
        tarfile.open(archive_path, "r:gz").extractall(path=datasets_folder)

    print("Check loading the 20 newsgroups files...")
    load_files(train_path, charset='latin1')
    load_files(test_path, charset='latin1')
    print("Success!")

if __name__ == "__main__":
    datasets_folder = get_datasets_folder()
    check_twenty_newsgroups(datasets_folder)
