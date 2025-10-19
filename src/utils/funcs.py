
import json


def load_json(path):
	d = {}
	with open(path, mode="r") as f:
		d = json.load(f)
	return d


