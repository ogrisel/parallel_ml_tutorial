# Makefile used to manage the git repository, not for the tutorial

check:
	python ipynbhelper.py --check

render:
	python ipynbhelper.py --render

clean:
	find . -name "*.pyc" | xargs rm -f
	python ipynbhelper.py --clean

clean-data:
	find . -name "*.pkl" | xargs rm -f
	find . -name "*.npy" | xargs rm -f
	find . -name "*.mmap" | xargs rm -f

all: check render clean
