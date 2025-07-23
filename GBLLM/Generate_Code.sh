# Scripts that adjust to the model and generate code
model_name=CodeLlama-34b-Instruct-hf
model_name=gemini-1.5-pro
model_name=gpt-3.5-turbo
model_name=gpt-4-1106-preview


api_key=''

base_data_path=../PIE/PIE_Python.csv
output_path=../output
role_key=english_generate_nl
prompt_key=english_generate_nl_io
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

# Adjustment parameters
python Merge_to_df_table_Python.py


base_data_path=../output/PIE__test__CodeLlama.csv
output_path=../output
role_key=english_triple_quotes
prompt_key=english_triple_quotes_cfg_io_use_nl
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

# Adjustment parameters
python Merge_to_df_table_Python.py

# Adjustment parameters
python PIE__Evaluate_code_execution_time.py