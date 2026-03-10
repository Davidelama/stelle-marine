#!/bin/bash
ns=(007 009 011 013) # (090)#
fs=(003 004 005 006)  #(010 030 060 100 150) # (020 060 120 200 300)#
rcs=(0.67)
ris=(4.0) # (4.0 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
mols=(001) #008
pass=(0.010 0.030 0.050 0.070 0.100 0.150 0.200)
suffixes=("" "_ctc" "_ctc_rll")
for n in "${ns[@]}";
do
for f in "${fs[@]}";
do
for rc in "${rcs[@]}";
do
for ri in "${ris[@]}";
do
for m in "${mols[@]}";
do
for pas in "${pass[@]}";
do
for suffix in "${suffixes[@]}";
do

name=mar_n"$n"_f"$f"_rc"$rc"_rb1.13_rcb0.57_pe2.3_rf10.0_Dt0.010_Dr0.800_gm100_dp"$pas"_rp0.333_Dp0.015"$suffix"; #_mols"$mols"_ri"$ri"
echo submitting $name;

cd ../../data/01_raw/stelle/"$name"/
chmod +x schain.sh
chmod +x schain_restart.sh
./schain.sh
cd ../../
done
done
done
done
done
done
done


