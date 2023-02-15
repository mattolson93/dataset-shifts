d3.json("static/cluster_data_1.json", function(d){
	console.log(d)

	const test_cluster_1 = d["test"][0]
		.filter((d, i) => i < 12);

	const test_cluster_2 = d["test"][1]
		.filter((d, i) => i < 12);

	const test_cluster_3 = d["test"][2]
		.filter((d, i) => i < 12);

	const test_cluster_4 = d["test"][3]
		.filter((d, i) => i < 12);

	var test = []
	for (i = 0; i < d["test"].length; i++) {
		for(j = 0; j < 12; j++){
			test.push(d["test"][i][j])
		}
		test.push("")	
	}

	d3.select('#cluster1')
		.selectAll("img")
		.data(test[0])
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})

	d3.select('#cluster2')
		.selectAll("img")
		.data(test_cluster_2)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})
	
	d3.select('#cluster3')
		.selectAll("img")
		.data(test_cluster_3)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})

	d3.select('#cluster4')
		.selectAll("img")
		.data(test_cluster_4)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})

	// training :))))
	const train_cluster_1 = d["train"][0]
		.filter((d, i) => i < 12);

	const train_cluster_2 = d["train"][1]
		.filter((d, i) => i < 12);

	const train_cluster_3 = d["train"][2]
		.filter((d, i) => i < 12);

	const train_cluster_4 = d["train"][3]
		.filter((d, i) => i < 12);

	
	d3.select('#train_cluster1')
		.selectAll("img")
		.data(train_cluster_1)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})
		.on("click", function(d){
			console.log("help")
		})

	d3.select('#train_cluster2')
		.selectAll("img")
		.data(train_cluster_2)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})
	
	d3.select('#train_cluster3')
		.selectAll("img")
		.data(train_cluster_3)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})

	d3.select('#train_cluster4')
		.selectAll("img")
		.data(train_cluster_4)
		.enter()
		.append("img")
		.attr("src", function(d){
			x = d.path.substring(1,d.path.length)
			test = '../static' + x;
			return test;
		})

})
    