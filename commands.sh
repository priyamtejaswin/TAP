# Eval
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --config configs/vqa/m4c_textvqa/tap_refine.yml --save_dir save/m4c_base_val --run_type val --resume_file save/finetuned/textvqa_tap_base_best.ckpt --evalai_inference true --verbose_dump true --gpu 1

# Scratch
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --config configs/vqa/m4c_textvqa/tap_ocrcc_sourceloss.yml --save_dir save/m4c_ocrcc_sourceloss --gpu 1

# Fine tune
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --config configs/vqa/m4c_textvqa/tap_ocrcc_sourceloss.yml --save_dir save/m4c_ocrcc_sourceloss --resume_file save/finetuned/textvqa_tap_ocrcc_best.ckpt --gpu 1

# Debug params
# ["--tasks", "vqa", "--datasets", "m4c_textvqa", "--model", "m4c_split", "--config", "configs/vqa/m4c_textvqa/tap_refine.yml", "--save_dir", "save/m4c_base_test", "--run_type", "val", "--resume_file", "save/finetuned/textvqa_tap_base_best.ckpt"]
# ["--tasks", "vqa", "--datasets", "m4c_textvqa", "--model", "m4c_split", "--config", "configs/vqa/m4c_textvqa/tap_ocrcc_sourceloss.yml", "--save_dir", "save/m4c_ocrcc_sourceloss", "--gpu", "1"]

# Generate data
python textvqa_source.py --truth TextVQA_0.5.1_train.json --features ../TAP/data/imdb/m4c_textvqa/imdb_train_ocr_en.npy --outpath ./imdb_train_ocr_with_source.npy
python textvqa_source.py --truth TextVQA_0.5.1_val.json --features ../TAP/data/imdb/m4c_textvqa/imdb_val_ocr_en.npy --outpath ./imdb_val_ocr_with_source.npy
