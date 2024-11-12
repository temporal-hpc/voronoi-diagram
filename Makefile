SHELL := /bin/bash # Use bash syntax
DEBUG=DEBUG
SAVE=SSTEP
PARAMS=-fopenmp ${ARCH}
ARCH=-arch sm_80
COMP= -Xcompiler 
ITER=SAVE_ITER
S1= JFA1	
S2= JFA2
all:
	nvcc $(COMP) $(PARAMS) -O3 test.cu -o test  -lm
cmp3d:
	nvcc $(COMP) $(PARAMS) -O3 test3d.cu -o test3d  -lm
save:
	nvcc $(COMP) $(PARAMS) -O3 -DSSTEP test.cu -o test -lm
test:
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
	./${1} ${2} ${3} 0 100 1 1 32 ${4} 1 1 ${5} 0 0 1
test_lj_sample:
	./test 5000 ${1} ${2} 10 1 1 32 0 1 1 4 1 25 1
test_uni_sample:
	./test 4000 ${1} ${2} 100 1 1 32 0 1 1 1 0 0 1
test_djfa_uni:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 1 0 0 1
test_djfa3d_uni:
	./test3d ${1} ${2} ${3} 100 1 1 16 0 1 1 1 0 0 1
test_rjfa_uni:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 ${4} 0 0 1
test_rjfa3d_uni:
	./test3d ${1} ${2} ${3} 100 1 1 16 0 1 1 ${4} 0 0 1
test_djfa_lj:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 1 1 ${4} 1
test_rjfa_lj:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 ${4} 1 ${5} 1
test_djfa_nb:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 1 2 ${4} 1
test_djfa3d_nb:
	./test3d ${1} ${2} ${3} 100 1 1 16 0 1 1 1 2 ${4} 1
test_rjfa_nb:
	./test ${1} ${2} ${3} 100 1 1 32 0 1 1 ${4} 2 ${5} 1
