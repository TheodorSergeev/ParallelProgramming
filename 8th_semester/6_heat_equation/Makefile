all: target clean run

target: heat.cpp
	mpiCC -std=c++11 -o main heat_ver2.cpp

clean:
	rm -rf *.o

run:
	for p_num in 1; do \
		echo "p = $$p_num"; \
		mpiexec -np $$p_num ./main; \
	done \