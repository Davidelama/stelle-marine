#!/bin/bash
NRESTARTS={{nrestarts}}
jobid=$(sbatch srestart.slrm | awk '{n=split($0,a,".");print(a[1])}')
echo "launched job $jobid"
for ((i=1;i<=${NRESTARTS};i++)) # total number of restarts, change it.
do
  jobidold=$jobid
  jobid=$(sbatch --dependency=afterok:${jobidold} srestart.slrm)
  echo "launched job $jobid -depending on $jobidold"
done
