#!/bin/bash
NRESTARTS={{nrestarts}}
jobid=$(qsub head.pbs | awk '{n=split($0,a,".");print(a[1])}')
echo "launched job $jobid"
for ((i=1;i<=${NRESTARTS};i++)) # total number of restarts, change it.
do
  jobidold=$jobid
  jobid=$(qsub -W depend=afterok:${jobidold} restart.pbs)
  echo "launched job $jobid -depending on $jobidold"
done
