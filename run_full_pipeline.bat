@echo off
echo ==========================================================
echo SINGLE INTERSECTION MO-CL-D3QN - FULL PIPELINE
echo ==========================================================
python single_intersection_experiment.py --mode full_pipeline --episodes 120 --eval-episodes 5 --sumocfg scenario/environment.sumocfg
python plot_results.py --csv outputs_single_intersection/evaluation_summary.csv
echo ==========================================================
echo SELESAI
echo Ringkasan: outputs_single_intersection\evaluation_summary.csv
echo Grafik   : outputs_single_intersection\plots
echo ==========================================================
pause
