## Lessons learnt in the project:
### Setup the project with `uv`
Create venv, pyproject.toml and setup.sh \
In the setup.sh we define the installation of the package based on backend for torch. \
### Structure the project
- FTI: Feature, Training Inference architecture
1. The _feature pipeline_ transforms your data into features & labels, which are stored and versioned in a feature store. The feature store will act as the central repository of your features.
2. The _training pipeline_ ingests a specific version of the features & labels from the feature store and outputs the trained model weights, which are stored and versioned inside a model registry.
3. The _inference pipeline_ uses a given version of the features from the feature store and downloads a specific version of the model from the model registry.
[Project idea: LLM twin](https://medium.com/decodingai/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f)
[FTI pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

- The configuration should be stated outside `src` for: environment-specific, user-editable, not packaged.
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
logging, dvc, mlflow, docker, testing, refactor
finetuning model for fitness

https://bhavishyapandit9.substack.com/p/building-multimodal-embeddings-a
https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip
https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/
