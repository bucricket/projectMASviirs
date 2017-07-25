##! /usr/bin/bash

ifort -c corr_parms.f90
ifort -c corr_arrays.f90 
ifort calc_offset_correction.f90 -o calc_offset_correction

string="main.f90 \
        wndo.f corr_parms.o corr_arrays.o \
    -o run_correction"

echo $string
ifort $string



