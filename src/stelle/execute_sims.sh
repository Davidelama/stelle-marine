#!/bin/bash
ns=(013) # (090)#
fs=(006)  #(010 030 060 100 150) # (020 060 120 200 300)#
rcs=(0.67)
mols=(002) #008
suffixes=("_pe2.3_rf10.0_bro0.8") #_gh_rf20.0 ("_gh_grft_star_stk_iden0.001_ieps3.0_irc0.75") #_bad_bfix("_opendef") #("_impl5.0") #("_star_impl") #("")
for n in "${ns[@]}";
do
for f in "${fs[@]}";
do
for rc in "${rcs[@]}";
do
for m in "${mols[@]}";
do
for suffix in "${suffixes[@]}";
do

name=mar_n"$n"_f"$f"_rc"$rc"_rb1.13_rcb0.57"$suffix""$al"; #_mols"$mols"
echo executing $name;
#mkdir ../../data/01_raw/cycles/"$name"
cd ../../data/01_raw/stelle/"$name"
mpirun -np 4 ../../../../bin/lmp_mpi_25 -v minimize 1 -v datafile "$name".dat  -in "$name".lmp
cd ../../../../src/stelle/
done
done
done
done
done


