# Language Translation

# Using MLFlow Model Runs Using SQL Instead of Local Directory
## Install dependencies
`pip3 install -r requirements.txt`

## Set up the artifact directory:
```
mkdir -p mlflow/artifacts
chmod -R 1777  mlflow
```

## Creating a Local MLFlow Server:
`mlflow ui`

## Run the server with the detached backend processes using SQLLite:
`mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.sqlite.db --default-artifact-root './mlflow/artifacts' </dev/null &>/dev/null &`
<!-- --host 0.0.0.0 -->

## Run your Language Translation Model Pipeline: 
`python3 index.py`

## Model Used:
- `google/flan-t5-base` - Language Translation