DEBUG= DEBUG
SAVE= SSTEP
PARAMS= -fopenmp
COMP= -Xcompiler
S1= JFA1
S2= JFA2
all:
	nvcc $(COMP) $(PARAMS) -O3 voronoi.cu -o prog  -lm
save:
	nvcc $(COMP) $(PARAMS) -D$(SAVE) -O3 voronoi.cu -o prog  -lm
test:
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}
	./prog ${1} ${2} ${3} ${4} ${5}