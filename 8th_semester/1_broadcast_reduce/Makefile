all: target clean

target: broadcast_reduce_timing.cpp
	mpiCC -std=c++11 -o main broadcast_reduce_timing.cpp

clean:
	rm -rf *.o