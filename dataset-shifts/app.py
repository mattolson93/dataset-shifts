from flask import Flask, render_template, redirect, url_for
import json

app = Flask(__name__)

umap_size = 400

@app.route("/")
def index():
	return redirect(url_for('scores'))

@app.route("/scores")
def scores():
	return render_template('scores.html', data=get_scores(count=500), umap_size=umap_size)

@app.route("/neighbors")
def neighbors():
	data = get_scores(count=500)
	for item in data["test"]:
		item["closest_evals"] = ",".join([str(x["id"]) for i, x in enumerate(item["closest_evals"]) if i < 5 or float(x["val"]) < 10.0])
		item["closest_trains"] = ",".join([str(x["id"]) for i, x in enumerate(item["closest_trains"]) if i < 5 or float(x["val"]) < 10.0])
	return render_template('neighbors.html', data=data, umap_size=umap_size)

@app.route("/clusters")
def clusters():
	return render_template('clusters.html', data=get_clusters(count=100), umap_size=umap_size)


# Loads the old index.html which uses cluster.js and script.js
@app.route("/index_debug")
def index_debug():
	return render_template('index.html')

@app.route("/results/<int:result_id>")
def get_ranking_results(result_id):
	return render_template('results.html', result_id=result_id)

@app.route("/data/<int:result_id>")
def get_data_for_result(result_id):
	with open("model_results/results_"+str(result_id)+".json", "r") as f:
		d = json.load(f)
		return json.dumps(d)

@app.route("/data_comparison_view/<int:result_id>/<int:instance_id>")
def get_data_for_given_instance(result_id, instance_id):
	# same as above or only for this image
	d = {"something": ["nothing", "nothing"]}
	return json.dumps(d)




RESULT_ID = 5

def get_scores(count):
	with open("model_results/results_"+str(RESULT_ID)+".json", "r") as content:
		data = json.load(content)

		test_data = [x for l in data["test"] for x in l]
		print(len(test_data), 'test')
		highest_test_data = sorted(test_data, key=lambda d: d["score"])[:count]
		highest_test_data = rescale_umap(highest_test_data)
		
		train_data = [x for l in data["train"] for x in l]
		#train_data = data["train"]
		print(len(train_data), 'train')
		highest_train_data = sorted(train_data, key=lambda d: d["score"])[:count]
		highest_train_data = rescale_umap(highest_train_data)
		
	return {'train': highest_train_data, 'test': highest_test_data}

def rescale_umap(highest_test_data):
	max_emb_x = max([x["emb_x"] for x in highest_test_data])
	min_emb_x = min([x["emb_x"] for x in highest_test_data])
	max_emb_y = max([x["emb_y"] for x in highest_test_data])
	min_emb_y = min([x["emb_y"] for x in highest_test_data])
	range_larger = max(max_emb_x - min_emb_x, max_emb_y - min_emb_y)
	for x in highest_test_data:
		x["emb_x"] = (x["emb_x"] - min_emb_x) / range_larger
		x["emb_y"] = (x["emb_y"] - min_emb_y) / range_larger
	return highest_test_data	

def sorted_filter(data, predicate, count = 1):
	filtered = list(filter(predicate, data))
	filtered = sorted(filtered, key=lambda d: d['score'])
	return filtered[:count]


def get_clusters(count):
	clusters = {'test':[], 'train':[], "flatted_train": [], "flatted_test": []}
	with open("model_results/results_"+str(RESULT_ID)+".json", "r") as content:
		data = json.load(content)
		test_data, train_data = data['test'], data['train']
		for c in test_data:
			clusters['test'].append(c[:count])
		for c in train_data:
			clusters['train'].append(c[:count])

		clusters["flatted_test"] = rescale_umap([x for l in clusters["test"] for x in l])
		clusters["flatted_train"] = rescale_umap([x for l in clusters["train"] for x in l])

	return clusters


if __name__ == "__main__":
	app.run()