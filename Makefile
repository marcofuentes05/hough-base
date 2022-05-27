all: pgm.o	houghBase

houghBase:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o houghBase

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
