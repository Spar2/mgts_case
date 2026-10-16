#!/bin/bash
# usage: scripts/run_resumable.sh <run_name> <command...>
# Relaunches <command> --resume auto (detached, setsid) if no live process for run and artifacts/ckpt/<run>/DONE absent.
# Trainers must save artifacts/ckpt/<run>/last.pt (model+opt+sched+step+RNG+best) every <=15 min and touch DONE at the end.
cd /workspace/dts_solution
run=$1; shift
d=artifacts/ckpt/$run; mkdir -p $d
[ -f $d/DONE ] && { echo "$run DONE"; exit 0; }
if [ -f $d/pid ] && kill -0 "$(cat $d/pid)" 2>/dev/null; then echo "$run alive pid $(cat $d/pid)"; exit 0; fi
echo "$@" > $d/cmd
setsid nohup bash -c "OMP_NUM_THREADS=2 $* --resume auto >> $d/log.txt 2>&1" >/dev/null 2>&1 < /dev/null &
echo $! > $d/pid
echo "$(date -u +%H:%M) RESUMABLE launch $run pid $! : $*" >> /workspace/run_logs/_events.log
