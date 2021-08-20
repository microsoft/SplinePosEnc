#!/bin/bash

python scripts/run_regress_sdf.py --alias 0406_armadillo_regress --name armadillo --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_armadillo_regress --name armadillo --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_armadillo_regress --name armadillo --epoch 400

python scripts/run_regress_sdf.py --alias 0406_bimba_regress --name bimba --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_bimba_regress --name bimba --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_bimba_regress --name bimba --epoch 400

python scripts/run_regress_sdf.py --alias 0406_bunny_regress --name bunny --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_bunny_regress --name bunny --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_bunny_regress --name bunny --epoch 400

python scripts/run_regress_sdf.py --alias 0406_dragon_regress --name dragon  --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dragon_regress --name dragon --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_dragon_regress --name dragon --epoch 400

python scripts/run_regress_sdf.py --alias 0406_dfaust_m_regress --name dfaust_m  --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dfaust_m_regress --name dfaust_m --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_dfaust_m_regress --name dfaust_m --epoch 400

python scripts/run_regress_sdf.py --alias 0406_dfaust_m_regress --name dfaust_f  --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dfaust_m_regress --name dfaust_m --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_dfaust_m_regress --name dfaust_m --epoch 400

python scripts/run_regress_sdf.py --alias 0406_fandisk_regress --name fandisk  --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_fandisk_regress --name fandisk --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_fandisk_regress --name fandisk --epoch 400

python scripts/run_regress_sdf.py --alias 0406_gargoyle_regress --name gargoyle  --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_gargoyle_regress --name gargoyle --epoch 400
python scripts/run_calc_sdf_err.py --alias 0406_gargoyle_regress --name gargoyle --epoch 400
