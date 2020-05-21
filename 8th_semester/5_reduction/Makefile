all: target clean

target: reduction.cpp
	mpiCC -std=c++11 -o main reduction.cpp

clean:
	rm -rf *.o
