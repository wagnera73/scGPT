#!/bin/bash
#SBATCH --job-name=nested_loop
#SBATCH --output=logs/nested_loop.out
#SBATCH --time=02:00:00

# Define the values for each of your 3 input fields
param1_vals=("save/scGPT_human" "save/scGPT_kidney")
param2_vals=("celltype" "condition" "celltype_condition")
param3_vals=("class_balance" "none")

counter=1

# Nested loops for n = 3
for p1 in "${param1_vals[@]}"; do
    for p2 in "${param2_vals[@]}"; do
        for p3 in "${param3_vals[@]}"; do

            # --balance_classes is a store_true flag (no value), so we pass
            # the flag token itself, or nothing at all. balance_flag is left
            # unquoted below so an empty value disappears from the argument
            # list instead of being passed as a stray "" argument.
            balance_flag=""
            if [ "$p3" == "class_balance" ]; then
                balance_flag="--balance_classes"
            fi

            echo "----------------------------------------"
            echo "Starting Test #$counter"
            echo "Parameters -> Param1: $p1 | Param2: $p2 | Param3: $p3"
            echo "----------------------------------------"

            bsub -q gpu-a100 -W 5:00 -n 4 -M 200000 -gpu "num=2" -R "span[hosts=1]" -o "./logs/%J.out" -e "./logs/%J.err" python scripts/main.py \
			  --dataset_name AML \
			  --model scGPT \
			  --task finetune \
			  --load_model $p1 \
			  --project scGPT-AML-celltype \
			  --label_column $p2 \
			  $balance_flag

            ((counter++))
            
        done
    done
done

echo "All 12 nested tests completed successfully."