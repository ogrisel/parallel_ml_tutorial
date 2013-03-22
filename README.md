# Parallel Machine Learning with scikit-learn and IPython


[![Video Tutorial](https://raw.github.com/ogrisel/parallel_ml_tutorial/master/resources/youtube_screenshot.png)](https://www.youtube.com/watch?v=iFkRt3BCctg)


## Scope of this tutorial:

- Learn about scalable feature extraction for text classification and
  clustering

- Learn how to perform parallel cross validation and hyper parameters grid
  search in parallel with IPython.

- Learn to analyze the kinds of common errors predictive models are subject to
  and how to refine your modeling to take this analysis into account.

- Learn to optimize memory allocation on your computing nodes with numpy memory
  mapping features.

- Learn how to run a cheap IPython cluster for interactive predictive modeling on
  the Amazon EC2 spot instances using [StarCluster](http://star.mit.edu/cluster/).


## Target audience

This tutorial targets developers with a prior experience with scikit-learn and
machine learning concepts in general.

It is recommended to first go through one of the tutorials hosted at
[scikit-learn.org](http://scikit-learn.org) if you are new to scikit-learn.

You might might also want to have a look at [SciPy Lecture
Notes](http://scipy-lectures.github.com) first if you are new to the NumPy /
SciPy / matplotlib ecosystem.


## Setup

Install NumPy, SciPy, matplotlib, IPython and scikit-learn in their latest
stable version (0.13.1 for IPython and 0.13.1 for scikit-learn at the time of
writing).

You can find up to date installation instructions on
[scikit-learn.org](http://scikit-learn.org) and
[ipython.org](http://ipython.org) .

To check your installation, launch the `ipython` interactive shell in a console
and type the following import statements to check each library:

    >>> import numpy
    >>> import scipy
    >>> import matplotlib
    >>> import sklearn

If you don't get any message, everything is fine. If you get an error message,
please ask for help on the mailing list of the matching project and don't
forget to mention the version of the library you are trying to install along
with the type of platform and version (e.g. Windows 8, Ubuntu 12.10, OSX
10.8...).

You can exit the `ipython` shell by typing `exit`.

## Fetching the data

It is recommended to fetch the datasets ahead of time before diving into the
tutorial material itself. To do so run the `fetch_data.py` script in this
folder:

    python fetch_data.py


## Using the IPython notebook to follow the tutorial

The tutorial material and exercises are hosted in a set of IPython executable
notebook files.

To run them interactively do:

    $ cd notebooks
    $ ipython notebook

To run the same notebooks along with the solutions to the inline exercises,
run instead:

    $ cd ipython solutions
    $ ipython notebook

This should automatically open a new browser window listing all the notebooks
of the folder.

You can then execute the cell in order by hitting the "SHIFT-ENTER" keys and
watch the output display directly under the cell and the cursor move on to the
next cell. Go to the "Help" menu for links to the notebook tutorial.


TODO: add links to online rendered versions as well
