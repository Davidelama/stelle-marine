#!/bin/bash
ns=(013) # (090)#
fs=(003)  #(010 030 060 100 150) # (020 060 120 200 300)#
rcs=(0.67)
ris=(0.0) #(4.0 5.0 6.0 7.0 8.0 9.0  10.0 12.0 14.0 16.0 18.0 20.0)
mols=(002) #008
suffixes=("_mols004_pe2.3_rf10.0_bro0.8")
for n in "${ns[@]}";
do
for f in "${fs[@]}";
do
for rc in "${rcs[@]}";
do
for ri in "${ris[@]}";
do
for suffix in "${suffixes[@]}";
do

name=mar_n"$n"_f"$f"_rc"$rc"_rb1.13_rcb0.57"$suffix" #_ri"$ri"; #_mols"$mols"
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


