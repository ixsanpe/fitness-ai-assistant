### Setup the project with `uv`
Create venv, pyproject.toml and setup.sh \
In the setup.sh we define the installation of the package based on backend for torch. \
We include the
### Build the dataset
JOSNLines files are easier to split and process in parallel (real-time, parallel processing)
The vector db comparison: https://docs.langchain.com/oss/python/integrations/vectorstores
### Multimodal embeddings: text and images
Ability to translate diverse data types into a common representational format in a high-dimensional space, where data of similar semantic are placed close together.
[Link info](https://milvus.io/docs/embeddings.md#Embedding-Overview)
#### CLIP 
CLIP Processor: includes the image processor and the tokenizer. The inputs are seet in TextKwargs 

### Model evaluation
Most multimodal models use zero-shot learning (ZSL) where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to. 

## pending steps:
logging, dvc, mlflow, docker
finetuning model for fitness

https://bhavishyapandit9.substack.com/p/building-multimodal-embeddings-a
https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip
https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/