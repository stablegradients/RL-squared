#!/bin/bash

# MuJoCo environments to test
ENVIRONMENTS=(
    "Reacher-v4"
    "Swimmer-v5"
    "Walker2d-v5"
    "InvertedPendulum-v5"
    "Pusher-v4"
    "Reacher-v4"
    "Swimmer-v5"
    "Walker2d-v5"
)

# Seeds to use
SEEDS=(42 43 44 45)

# Number of GPUs and jobs per GPU
NUM_GPUS=4
JOBS_PER_GPU=4

# Create logs directory
mkdir -p logs
mkdir -p checkpoints

# Function to get next available GPU slot
get_next_slot() {
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if [ ${SLOTS_USED[$i]} -lt $JOBS_PER_GPU ]; then
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
declare -A GPU_SLOTS

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
        checkpoint_dir="checkpoints/${exp_name}"
        
        echo "Starting experiment: $exp_name on GPU $SLOT"
        
        # Run the experiment with original Haarnoja SAC parameters
        CUDA_VISIBLE_DEVICES=$SLOT python main.py \
            --env_name "$env" \
            --seed "$seed" \
            --policy_type gaussian \
            --reparameterize \
            --hidden_dim 256 \
            --lr 3e-4 \
            --gamma 0.99 \
            --tau 0.005 \
            --alpha 0.2 \
            --auto_tune_alpha True \
            --batch_size 256 \
            --buffer_size 1000000 \
            --wandb_project "sac-ablation" \
            --tags "original-sac,$env" \
            --checkpoint_dir "$checkpoint_dir" > "$log_file" 2>&1 &
        
        # Store the PID and its GPU slot
        pid=$!
        PIDS+=($pid)
        GPU_SLOTS[$pid]=$SLOT
        
        echo "Started job with PID $pid on GPU $SLOT (Slot usage: ${SLOTS_USED[$SLOT]}/$JOBS_PER_GPU)"
    done
done

# Wait for all remaining jobs to finish
echo "All jobs submitted. Waiting for completion..."
wait

echo "All experiments completed!"