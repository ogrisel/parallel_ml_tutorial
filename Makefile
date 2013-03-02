clean-pyc:
	find . -name "*.pyc" | xargs rm

clean-data:
	find . -name "*.pkl" | xargs rm
	find . -name "*.npy" | xargs rm
	find . -name "*.mmap" | xargs rm

clean-notebooks:
	python housekeeping.py clean notebooks
	python housekeeping.py clean solutions

clean: clean-pyc clean-data clean-notebooks
