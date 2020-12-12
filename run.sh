python convert_mimic.py
python profeat_mimic.py --input_json data/dataset_mimic_rm.json --output_dir data/mimicrm --images_root /media/hdd/donghao/imcaption/R2Gen/data/mimic_cxr/images
python scripts/prepro_labels.py --input_json data/dataset_mimic_rm.json --output_json data/mimicrmtalk.json --output_h5 data/mimicrmtalk --max_length 60
python scripts/prepro_reference_json_mimic.py --input_json data/mimicrmtalk.json --output_json data/mimicrm_eval.json
python scripts/prepro_reference_json_mimic.py --input_json data/dataset_mimic.json --output_json data/mimic_eval.json
# modify  
python tools/train.py --cfg configs/mimic_aoa.yml --id aoa
python tools/train.py --cfg configs/mimic_a2i2.yml --id a2i2
python tools/train.py --cfg configs/fds_tf.yml --id tfdebugv3
python tools/train.py --cfg configs/mimic_fc.yml --id fc
python tools/train.py --cfg configs/mimic_m2.yml --id m2
python tools/train.py --cfg configs/mimic_updown.yml --id updown
python tools/train.py --cfg configs/mimic_bert.yml --id bert
python tools/train.py --cfg configs/mimicrm_m2.yml --id m2rm
python tools/train.py --cfg configs/mimicrm_tf.yml --id tfrm
python tools/eval_mimic.py --model logs/mimic_updown/model-best.pth --infos_path logs/mimic_updown/infos_updown-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_tfv2/model-best.pth --infos_path logs/mimic_tfv2/infos_mimic_tf-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_a2i2/model-best.pth --infos_path logs/mimic_a2i2/infos_a2i2-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_fc/model-best.pth --infos_path logs/mimic_fc/infos_fc-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_m2/model-best.pth --infos_path logs/mimic_m2/infos_m2-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_aoa/model-best.pth --infos_path logs/mimic_aoa/infos_aoa-best.pkl --split test
