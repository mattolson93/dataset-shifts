
d3.json("static/results_3.json", function(d) {  
    const dataset = d;

    const highest_data = dataset
        .filter(d => d.datasettype == "train")
        .sort((a, b) => (a.id > b.id) ? 1 : -1)
        .filter((d, i) => i < 1000);

    const highest_test_data = dataset
        .filter(d => d.datasettype != "train")
        .sort((a, b) => (a.id  > b.id) ? 1 : -1)
        .filter((d, i) => i < 1000);


    d3.select('#score_box')
        .selectAll("img")
        .data(highest_data)
        .enter()
        .append("img")
        .attr("id", function(d){
            x = "score-id-" + d["id"]
            return x;
        })
        .attr("src", function(d){
            x = d.path.substring(1,d.path.length)
            test = '../static' + x;
            //why doesn't this work
            return test;
        })
        .on("click", function(d){
            //undo previous click
            d3.select("#img_selected")
                .html("")
            d3.select("#img_selected_prop")
                .html("")
            d3.selectAll(".clicked")
                .classed("clicked", false)
            d3.select('#density_sim_train')
                .html("")
            d3.select('#density_sim_test')
                .html("")
            d3.select('#orig_sim_train')
                .html("")
            d3.select('#orig_sim_test')
                .html("")
            
            
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "green")
                .attr("r", "5")


            //applying click attribute				
            d3.select(this).classed("clicked", true);
            d3.select("#img_selected")
                .append("img")
                .attr("src", function(d_this){
                    x = d.path.substring(1,d.path.length)
                    test = '../static' + x;

                    //why doesn't this work
                    return test;
                })
            d3.select("#img_selected_prop")
                .html(function(v){
                    id = "ID: " + d["id"] + "<br>";
                    datatype = "Data Type: " + d["datasettype"] + "<br>"
                    label = "Label: " + d["label"] + "<br>"

                    return id + datatype + label;
                })

            //appending similar images!
            sim_train = d["spaces"][0]["closest_trains"]
            d3.select('#density_sim_train')
                .selectAll("img")
                .data(sim_train)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
            //appending similar images!
            sim_test = d["spaces"][0]["closest_evals"]
            d3.select('#density_sim_test')
                .selectAll("img")
                .data(sim_test)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })

            og_train = d["spaces"][1]["closest_trains"]
            d3.select('#orig_sim_train')
                .selectAll("img")
                .data(og_train)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
            //appending similar images!
            og_test = d["spaces"][1]["closest_evals"]
            d3.select('#orig_sim_test')
                .selectAll("img")
                .data(og_test)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
        })
        .on("mouseover", function(d){
            d3.select(this).classed("hover", true)
            d3.selectAll(`.imagedot-${d.id}`)
                .attr("stroke-width", "10")
                .attr("opacity", 1.5);
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "yellow")
                .attr("r", "5")

        })
        .on("mouseout", function(d){
            d3.selectAll(".hover")
                .classed("hover", false)
            d3.selectAll(`.imagedot-${d.id}`)
                .attr("stroke-width", 0)
                .attr("opacity", .2);
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "black")
                .attr("r", "2")
            
        })


    
        d3.select('#score_outlier_box')
        .selectAll("img")
        .data(highest_test_data)
        .enter()
        .append("img")
        .attr("id", function(d){
            x = "score-id-" + d["id"]
            return x;
        })
        .attr("src", function(d){
            x = d.path.substring(1,d.path.length)
            test = '../static' + x;
            //why doesn't this work
            return test;
        })
        .on("click", function(d){
            //undo previous click
            d3.select("#img_selected")
                .html("")
            d3.select("#img_selected_prop")
                .html("")
            d3.selectAll(".clicked")
                .classed("clicked", false)
            d3.select('#density_sim_train')
                .html("")
            d3.select('#density_sim_test')
                .html("")
            d3.select('#orig_sim_train')
                .html("")
            d3.select('#orig_sim_test')
                .html("")
            
            
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "green")
                .attr("r", "5")


            //applying click attribute				
            d3.select(this).classed("clicked", true);
            d3.select("#img_selected")
                .append("img")
                .attr("src", function(d_this){
                    x = d.path.substring(1,d.path.length)
                    test = '../static' + x;

                    //why doesn't this work
                    return test;
                })
            d3.select("#img_selected_prop")
                .html(function(v){
                    id = "ID: " + d["id"] + "<br>";
                    datatype = "Data Type: " + d["datasettype"] + "<br>"
                    label = "Label: " + d["label"] + "<br>"

                    return id + datatype + label;
                })

            //appending similar images!
            sim_train = d["spaces"][0]["closest_trains"]
            d3.select('#density_sim_train')
                .selectAll("img")
                .data(sim_train)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
            //appending similar images!
            sim_test = d["spaces"][0]["closest_evals"]
            d3.select('#density_sim_test')
                .selectAll("img")
                .data(sim_test)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })

            og_train = d["spaces"][1]["closest_trains"]
            d3.select('#orig_sim_train')
                .selectAll("img")
                .data(og_train)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
            //appending similar images!
            og_test = d["spaces"][1]["closest_evals"]
            d3.select('#orig_sim_test')
                .selectAll("img")
                .data(og_test)
                .enter()
                .append("img")
                .attr("class", function(v){
                    x = dataset[v]["id"];
                    y = 'imagedot-' + x;

                    return 'imagedot ' + y;
                })
                .attr("src", function(v){
                    w = dataset[v]["path"]
                    x = w.substring(1,v.length)
                    test = '../static' + x;
                    return test;

                })
                .on("mouseover", function(w){
                    d3.selectAll(`.imagedot-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);

                    d3.select(this).classed("hover", true)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", "10")
                        .attr("opacity", 1.5);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "orange")
                        .attr("r", "5")
                })
                .on("mouseout", function(w){
                    d3.selectAll(".hover")
                        .classed("hover", false)
                    d3.selectAll(`.score-id-${w}`)
                        .attr("stroke-width", 0)
                        .attr("opacity", .2);
                    d3.selectAll(`.dot-${w}`)
                        .attr("fill", "black")
                        .attr("r", "2")
                    
                })
        })
        .on("mouseover", function(d){
            d3.select(this).classed("hover", true)
            d3.selectAll(`.imagedot-${d.id}`)
                .attr("stroke-width", "10")
                .attr("opacity", 1.5);
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "yellow")
                .attr("r", "5")

        })
        .on("mouseout", function(d){
            d3.selectAll(".hover")
                .classed("hover", false)
            d3.selectAll(`.imagedot-${d.id}`)
                .attr("stroke-width", 0)
                .attr("opacity", .2);
            d3.selectAll(`.dot-${d.id}`)
                .attr("fill", "black")
                .attr("r", "2")
            
        })


    var umapdensity_svgContainer = d3.select("#umap_density")
        .append("svg")
        .attr("width", 1000)
        .attr("height", 1000);

    var circle = umapdensity_svgContainer.selectAll("circle")
        .data(highest_data)
        .enter()
        .append("circle");

    circle
        .attr("cx", function(d) {
            emb_x = d["spaces"][0]["emb_x"]
            return 250 + 35*emb_x;
        })
        .attr("cy", function(d) {
            emb_y = d["spaces"][0]["emb_y"]
            return 300 + 35*emb_y;
        })
        .attr("r", function(d) {
            return 3;
        })
        .attr("opacity", function(d) {
            // return Math.abs(d["spaces"][0]["score"]);
            return 1;
        })
        .attr("fill", function(d) {
            return "red";
        })
        .attr("class", v => `imagedot dot-${v.id}`)


    var umaporiginal_svgContainer = d3.select("#umap_orig")
        .append("svg")
        .attr("width", 1000)
        .attr("height", 1000);

    var circle = umaporiginal_svgContainer.selectAll("circle")
        .data(highest_data)
        .enter()
        .append("circle");

    circle
        .attr("cx", function(d) {
            emb_x = d["spaces"][1]["emb_x"]
            return 250 + 35*emb_x;
        })
        .attr("cy", function(d) {
            emb_y = d["spaces"][1]["emb_y"]
            return 300 + 35*emb_y;
        })
        .attr("r", function(d) {
            return 3;
        })
        .attr("fill", function(d) {
            return "black";
        })
        .attr("opacity", function(d) {
            return d["spaces"][1]["score"];
        })
        .attr("class", v => `imagedot dot-${v.id}`)


});        