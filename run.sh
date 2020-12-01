python tools/train.py --cfg configs/fds_tf.yml --id tfdebugv3
python tools/train.py --cfg configs/mimic_a2i2.yml --id a2i2
python tools/train.py --cfg configs/mimic_fc.yml --id fc
python tools/eval_mimic.py --model logs/mimic_tfv2/model-best.pth --infos_path logs/mimic_tfv2/infos_mimic_tf-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_a2i2/model-best.pth --infos_path logs/mimic_a2i2/infos_a2i2-best.pkl --split test
python tools/eval_mimic.py --model logs/mimic_fc/model-best.pth --infos_path logs/mimic_fc/infos_fc-best.pkl --split test