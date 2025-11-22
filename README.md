# fitness-ai-assistant
In this project I want to build a tool to help me assist with my gym and fitness training.
First, I want to build my own vector db to store the data and play with embeddings for multi modal dataset.
Later stages, I would like to create a recommender system or assitant to track my training data and check how I could improve the training.
## what you need for developer environment
- `pipx install uv`
#todo: ruff, docker
## start the project locallz
`chmod +x infra/setup-local.sh`
`./infra/setup-local.sh` will initialize the project with `uv` and install the correct version of `pytorch` depending on the backend. It will also retrieve the dataset to work locallz
`source .venv/bin/activate` will start the virtual environment we have just created.
## prepare a dataset of gym exercises
Using the dataset from `https://github.com/wrkout/exercises.json#`
Create embeddings and vector db

