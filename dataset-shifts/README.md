# dataset-shifts


## Installation
1. Clone the repo (fork)
```sh
git clone https://github.com/div-lab/dataset-shifts.git
```
2. Create a virtual environment
```sh
cd dataset-shifts && mkdir env && python3 -m venv env/
```
3. Activate the environment
```sh
. env/bin/activate
```
4. Install requirements
```sh
pip3 install -r requirements.txt
```


### Client

1. Run the python server
```sh
python3 -m http.server
```

Point your browser to http://0.0.0.0:8000/html/cluster.html


### Server Deployment (Dev)

1. Install requirements
```sh
pip3 install -r requirements.txt
```
2. Set environment variables
```sh
export FLASK_APP=server && export FLASK_ENV=development
```
3. Start development server
```sh
flask run
```

Point your browser to http://127.0.0.1:5000/


Use test users:
1. email: test@test.com     password: test
2. email: test2@test.com    password: test2
   
The server provides a web interface for creating new users at http://127.0.0.1:5000/signup.
We can write a script to add users to the db directly and disable this before deploying.

To force logout, go to http://127.0.0.1:5000/logout

### TODO Server Deployment (Https + Flip.engr)
