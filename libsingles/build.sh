#!/bin/bash

gcc -c -fpic -Ofast -march=native singles.c
gcc -shared -o libsingles.so singles.o

gcc -Ofast -march=native main.c -o main \
    -Wl,-rpath,/home/aaron/pet-insert-processing/libsingles \
    -L/home/aaron/pet-insert-processing/libsingles \
    -lsingles -lm
