#!/bin/bash

gcc -c -fpic -O3 -march=native singles.c
gcc -shared -o libsingles.so singles.o

gcc -O3 -march=native main.c -o main \
    -Wl,-rpath,/home/aaron/pet-insert-processing/libsingles \
    -L/home/aaron/pet-insert-processing/libsingles \
    -lsingles -lm
