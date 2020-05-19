all: target clean run

target: merge_sort.cpp
	mpiCC -std=c++11 -o main merge_sort.cpp

clean:
	rm -rf *.o
	rm -f merge_sort_time.txt

run:
	for p_num in 1 2 4 6 8 10 12 14 16 18 20 22 24; do \
		echo "p = $$p_num"; \
		mpiexec -np $$p_num ./main; \
	done \