#!/bin/bash

algorithms=("aerial" "aerial_no_att" "aerial_raw_history1" "qplex" "cw_qmix" "ow_qmix" "qmix" "mappo")
smac_maps=("3s5z" "10m_vs_11m" "2c_vs_64zg" "3s_vs_5z" "5m_vs_6m" "3s5z_vs_3s6z")

# Dec-Tiger Experiment (State-of-the-Art Comparison)
for i in `seq 1 50`;
do
  for algorithm in "${algorithms[@]}"
  do
    python3 src/main.py --config=$algorithm --env-config=dec_tiger with env_args.map_name=dec_tiger
  done
done

# Original SMAC Experiment (State-of-the-Art Comparison)
for i in `seq 1 20`;
do
  for algorithm in "${algorithms[@]}"
  do
    for smac_map in "${smac_maps[@]}"
    do
      python3 src/main.py --config=$algorithm --env-config=sc2 with env_args.map_name=$smac_map
    done  
  done
done

# MessySMAC Experiment (State-of-the-Art Comparison)
for i in `seq 1 20`;
do
  for algorithm in "${algorithms[@]}"
  do
    for smac_map in "${smac_maps[@]}"
    do
      python3 src/main.py --config=$algorithm --env-config=messy_sc2 with env_args.map_name=$smac_map
    done  
  done
done

# MessySMAC Experiment (State Uncertainty Robustness)

phis=("0.0" "0.05" "0.1" "0.15" "0.2")
Ks=("0" "5" "10" "15")

for i in `seq 1 20`;
do
  for algorithm in "${algorithms[@]}"
  do
    for smac_map in "${smac_maps[@]}"
    do
      for phi in "${phis[@]}" # Robustness against observation stochasticity
      do
        python3 src/main.py --config=$algorithm --env-config=messy_sc2 with env_args.map_name=$smac_map env_args.failure_obs_prob=$phi
      done
      for K in "${Ks[@]}" # Robustness against variance in initial states
      do
        python3 src/main.py --config=$algorithm --env-config=messy_sc2 with env_args.map_name=$smac_map env_args.randomize_initial_state=$K
      done
    done  
  done
done