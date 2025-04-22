# Fine Tuning Text-to-SQL Model

# Using MLFlow Model Runs Using SQL Insrtead of Local Directory
## Install dependencies
`pip3 install -r requirements.txt`

## Set up the artifact directory
```
mkdir -p mlflow/artifacts
chmod -R 1777  mlflow
```

## Run the server with the detacted process using SQLLite:
`mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.sqlite.db --default-artifact-root './mlflow/artifacts' </dev/null &>/dev/null &`

## Run your Model Training 
`python3 index.py`