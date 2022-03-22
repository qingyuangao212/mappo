model_names="fineTuneCA_15_6_1"
run=20

for model_name in $model_names; do
  echo $model_name
  CUDA_VISIBLE_DEVICES=1 python3 load_model.py --model_name $model_name --run $run
done