#!/bin/bash
ns=(005 007 009 011 013 015) # (090)#
fs=(003 004 005 006)  #(010 030 060 100 150) # (020 060 120 200 300)#
rcs=(0.67)
ris=(4.0) # (4.0 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
mols=(001) #008
suffixes=("_pe2.3_rf10.0_Dt0.010_Dr0.800_gm100")
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
# for seed in "${seeds[@]}";
# do

name=mar_n"$n"_f"$f"_rc"$rc"_rb1.13_rcb0.57"$suffix"; #_mols"$mols"_ri"$ri"
echo downloading $name;
#mkdir ../../data/01_raw/clicks
scp -i ~/.ssh/id_cineca -r dbreoni0@login.leonardo.cineca.it:~/stelle-marine/data/01_raw/stelle/"$name" ../../data/01_raw/stelle/
            done
        done
    done
done
done


