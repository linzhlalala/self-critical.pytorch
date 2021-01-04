import json
# retina_file_name = 'data/retinatinytalk.json'
retina_file_name = 'data/retina_tiny_eval.json'
with open(retina_file_name) as retina_json_file:
	retina_dataset = json.load(retina_json_file)
	print(retina_dataset.keys())