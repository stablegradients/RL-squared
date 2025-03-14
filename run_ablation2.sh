#!/bin/bash

# MuJoCo environments to test
ENVIRONMENTS=(
    "InvertedPendulum-v5"
    "Pusher-v4"
    "Reacher-v4"
    "Swimmer-v5"
    "Walker2d-v5"
)

# Seeds to use
SEEDS=(42 43 44 45 46 47 48 49)

# Number of GPUs
NUM_GPUS=4

# Create logs directory
mkdir -p logs

# Function to get next available GPU slot
get_next_slot() {
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if [ ${SLOTS_USED[$i]} -lt 2 ]; then
            echo $i
            return
        fi
    done
    echo "-1"  # No available slots
}

# Initialize slots counter
declare -a SLOTS_USED
for i in $(seq 0 $((NUM_GPUS-1))); do
    SLOTS_USED[$i]=0
done

# Track all running processes
declare -a PIDS

# Run all environment-seed combinations
for env in "${ENVIRONMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Wait for an available GPU slot
        while true; do
            SLOT=$(get_next_slot)
            if [ "$SLOT" != "-1" ]; then
                break
            fi
            
            # Check if any jobs have finished
            for i in "${!PIDS[@]}"; do
                if ! ps -p ${PIDS[$i]} > /dev/null; then
                    # Process has finished, free up its slot
                    SLOTS_USED[${GPU_SLOTS[$i]}]=$((SLOTS_USED[${GPU_SLOTS[$i]}]-1))
                    unset PIDS[$i]
                    unset GPU_SLOTS[$i]
                fi
            done
            
            # Wait a bit before checking again
            sleep 5
        done
        
        # Increment slot usage
        SLOTS_USED[$SLOT]=$((SLOTS_USED[$SLOT]+1))
        
        # Create a unique experiment name
        exp_name="${env}_seed${seed}"
        log_file="logs/${exp_name}.log"
        
        echo "Starting experiment: $exp_name on GPU $SLOT"
        
        # Run the experiment in the background
        CUDA_VISIBLE_DEVICES=$SLOT python main.py \
            --env_name "$env" \
            --seed "$seed" \
            --wandb_project "sac-ablation" \
            --tags "ablation,$env" \
            --checkpoint_dir "checkpoints/$exp_name" > "$log_file" 2>&1 &
        
        # Store the PID and its GPU slot
        pid=$!
        PIDS+=($pid)
        GPU_SLOTS[$pid]=$SLOT
        
        echo "Started job with PID $pid on GPU $SLOT"
    done
done

# Wait for all remaining jobs to finish
echo "All jobs submitted. Waiting for completion..."
wait

echo "All experiments completed!" 