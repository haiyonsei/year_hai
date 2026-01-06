
#MODEL=./transformer_ar_inverse_models/ar10x10-L18-d192-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr0.0005-bs128-seed42/best_model.pth 
MODEL=./transformer_ar_inverse_models/ar10x10-L15-d512-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr0.0001-bs128-seed42/best_model.pth

# Load할 모델의 num_layers 와 d_model에 따라 아래 설정을 바꾸어 주세요.
# beam search 와 sampling 방식이 있으며, decode_method 에서 beam 을 설정하는 경우에는 아래와 같이 셋팅하는 경우 400개의 beam 중에 best 200개를 고르게 됩니다.
# sampling 방식의 경우 beam_size는 무시됩니다.
python3 ./inverse_from_csv_10x10.py --csv_path ../specs/desired/noise_mask/noise_mask_single.csv  --model_path $MODEL --num_layers 15 --d_model 512 --num_candidates 200 --decode_method sampling --beam_size 400
python3 ./inverse_from_csv_10x10.py --csv_path ../specs/desired/noise_mask/noise_mask_dual.csv  --model_path $MODEL --num_layers 15 --d_model 512 --num_candidates 200 --decode_method sampling --beam_size 400

