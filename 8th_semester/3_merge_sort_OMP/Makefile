all: target clean

target: merge_sort.cpp
	gcc merge_sort.cpp -fopenmp -lstdc++ -lm -std=c++11 -o main 

clean:
	rm -rf *.o