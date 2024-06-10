SHELL := /bin/bash # Use bash syntax
DEBUG=DEBUG
SAVE=SSTEP
PARAMS=-fopenmp ${ARCH}
ARCH=-arch sm_80
COMP= -Xcompiler 
ITER=SAVE_ITER
S1= JFA1	
S2= JFA2
PATF= srun --container-name=cuda-11.2.2
PATF2= srun -p cpu
GPU= --gres=gpu:A100:1
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
target:
	iter=1; while [[$$iter -le 10]] ; do \
		make test 1=${1} 2=${2} 3=500 4=${3} 5=${4}; \
		((iter = iter + 1)) ; \
	done
	seeds=1000 ; while [[ $$number -le 20000 ]] ; do \
		iter = 1; while [[$$iter -le 10]] ; do \
	    	make test 1=${1} 2=${2} 3=$$iter 4=${3} 5=${4}; \
			((iter = iter + 1)) ; \
		done
	    ((number = number + 1000)) ; \
	done
metrics:
	number=1 ; while [[ $$number -le 10]] ; do \
			make test 1=${1} 2=${2}
			((number = number + 1)) ; \
	done
batch:
	make metrics 1=${1} 2=${2} 
pat_test:
	make target 1=${1} 2=1000 3=${2} 4=${3}
	make target 1=${1} 2=2000 3=${2} 4=${3}
	make target 1=${1} 2=4000 3=${2} 4=${3}
	make target 1=${1} 2=5000 3=${2} 4=${3}
	make target 1=${1} 2=6000 3=${2} 4=${3}
	make target 1=${1} 2=8000 3=${2} 4=${3}
