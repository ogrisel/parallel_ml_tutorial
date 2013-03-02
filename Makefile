clean:
	find . -name "*.pkl" | xargs rm
	find . -name "*.npy" | xargs rm
	find . -name "*.mmap" | xargs rm
