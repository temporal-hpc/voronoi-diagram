SHELL := /bin/bash # Use bash syntax
DEBUG= DEBUG
SAVE= SSTEP
PARAMS= -fopenmp
COMP= -Xcompiler
ITER=SAVE_ITER
S1= JFA1	
S2= JFA2
PATF= srun --container-name=cuda-11.2.2
PATF2= srun -p cpu
GPU= --gres=gpu:A100:1
all:
	nvcc $(COMP) $(PARAMS) -O3 voronoi.cu -o prog  -lm
save:
	$(PATF) nvcc $(COMP) $(PARAMS) -D$(SAVE) -O3 voronoi.cu -o prog  -lm
iters:
	$(PATF) nvcc $(COMP) $(PARAMS) -D$(SAVE) -D$(ITER) -O3 voronoi.cu -o prog  -lm
iters-own:
	nvcc $(COMP) $(PARAMS) -D$(SAVE) -D$(ITER) -O3 voronoi.cu -o prog  -lm
test:
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
	./prog ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
target:
	number=100 ; while [[ $$number -le 10000 ]] ; do \
	       make test 1=${1} 2=$$number 3=2 4=100 5=0 6=1 7=32 8=${2}; \
	       ((number = number + 100)) ; \
	done
pat_test:
	make target 1=1000 2=1
	make target 1=2000 2=1
	make target 1=4000 2=1
	make target 1=10000 2=1
	make target 1=20000 2=1
	#make target 1=40000 2=1