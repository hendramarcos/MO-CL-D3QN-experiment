@echo off
echo ==========================================================
echo DEPLOY SEMUA MODEL KE SUMO GUI
echo ==========================================================
python deploy_all_models_gui.py --method all --gui --wait-before-start --keep-open --sumocfg scenario/environment.sumocfg
echo ==========================================================
echo SELESAI
echo Ringkasan: outputs_deployment\deployment_comparison_summary.csv
echo ==========================================================
pause
