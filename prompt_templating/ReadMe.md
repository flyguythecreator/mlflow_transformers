# Prompt Templating with MLFLow and Huggingface Transformers

# Using MLFlow Model Runs Using SQL Instead of Local Directory
## Install dependencies
`pip3 install -r requirements.txt`

## Set up the artifact directory:
```
mkdir -p mlflow/artifacts
chmod -R 1777  mlflow
```

## Creating a Local MLFlow Server:
If you do not have a running mlflow server. Run the following command: 
- `mlflow ui`

## Run the server with the detached backend processes using SQLLite:
`mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.sqlite.db --default-artifact-root './mlflow/artifacts' </dev/null &>/dev/null &`
<!-- --host 0.0.0.0 -->

## Run your Language Translation Model Pipeline: 
`python3 index.py`

## Model Used:
- `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` - Text Generation