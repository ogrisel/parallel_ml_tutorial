clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-data:
	find . -name "*.pkl" | xargs rm -f
	find . -name "*.npy" | xargs rm -f
	find . -name "*.mmap" | xargs rm -f

strip:
	# Strip output and prompt numbers in solutions
	python housekeeping.py clean solutions

exercises:
	python housekeeping.py exercises


clean: clean-pyc clean-data strip

pre-commit: strip exercises

dist: pre-commit
	python fetch_data.py
