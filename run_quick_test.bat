@echo off
echo ==========================================================
echo SINGLE INTERSECTION MO-CL-D3QN - QUICK TEST
echo ==========================================================
python single_intersection_experiment.py --mode full_pipeline --episodes 5 --eval-episodes 2 --max-steps 500 --sumocfg scenario/environment.sumocfg
python plot_results.py --csv outputs_single_intersection/evaluation_summary.csv
echo ==========================================================
echo QUICK TEST SELESAI
echo ==========================================================
pause
