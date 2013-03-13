clean-pyc:
	find . -name "*.pyc" | xargs rm

clean-data:
	find . -name "*.pkl" | xargs rm
	find . -name "*.npy" | xargs rm
	find . -name "*.mmap" | xargs rm

strip:
	# Strip output and prompt numbers in solutions
	python housekeeping.py clean solutions

exercises:
	python housekeeping.py exercises


clean: clean-pyc clean-data strip

pre-commit: strip exercises

dist: pre-commit
	python fetch_data.py
