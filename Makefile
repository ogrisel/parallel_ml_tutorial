clean-pyc:
	find . -name "*.pyc" | xargs rm

clean-data:
	find . -name "*.pkl" | xargs rm
	find . -name "*.npy" | xargs rm
	find . -name "*.mmap" | xargs rm

clean-solutions:
	python housekeeping.py clean solutions

generate-exercises:
	python housekeeping.py exercises


clean: clean-pyc clean-data clean-notebooks

pre-commit: clean-solutions generate-exercises
