# fitness-ai-assistant (WIP)
In this project I want to build a tool to help me assist with my gym and fitness training.
First, I want to build my own vector db to store the data and play with embeddings for multi modal dataset.
Later stages, I would like to create a recommender system or assitant to track my training data and check how I could improve the training.

## project structure
#TODO
- FTI architecture, configs, tests

## what you need for developer environment
- `pipx install uv`
#TODO: wsl, python, docker, cicd pipelines

## start the project locally
`chmod +x infra/setup-local.sh`
`./infra/setup-local.sh` will initialize the project with `uv` and install the correct version of `pytorch` depending on the backend. It will also retrieve the dataset to work locally.
`source .venv/bin/activate` will start the virtual environment we have just created.

## prepare a dataset of gym exercises
Using the dataset from `https://github.com/wrkout/exercises.json#`, we download a set of folders where we include the exrecises pictures and information.
1. Build a dataset in `jsonl` to summarize all the exercises: stored in `processed`
2. Extract embbedings for each sample: we support 2 embedddings type: `sentence` and `CLIP`: stored in `processed/embeddings`
3. Create a vector DB where we store the embeddings with `milvus`

## model evaluation and selection
Create embeddings and vector db
