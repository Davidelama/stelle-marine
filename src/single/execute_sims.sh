#!/bin/bash
suffixes=("_pe10.0_bro") #_gh_rf20.0 ("_gh_grft_star_stk_iden0.001_ieps3.0_irc0.75") #_bad_bfix("_opendef") #("_impl5.0") #("_star_impl") #("")
for suffix in "${suffixes[@]}";
do

name=sin"$suffix""$al"; #_mols"$mols"
echo executing $name;
#mkdir ../../data/01_raw/cycles/"$name"
cd ../../data/01_raw/single/"$name"
mpirun -np 4 ../../../../bin/lmp_mpi_25 -v minimize 1 -v datafile "$name".dat  -in "$name".lmp
cd ../../../../src/single/
done



