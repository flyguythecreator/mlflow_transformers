import warnings
import logging
import os
import mlflow
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  pipeline,
)

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

# region Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger()

# Load the "sms_spam" dataset.
sms_dataset = load_dataset("sms_spam")
# logger.info(sms_dataset)

# Split train/test by an 8/2 ratio.
sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
train_dataset = sms_train_test["train"]
test_dataset = sms_train_test["test"]

# Print the Prepared Datasets
# logger.info("Training Dataset: ", train_dataset)
# logger.info("Training Dataset Data: ", train_dataset[:5])
# logger.info("Test Dataset: ", test_dataset)
# logger.info("SSM Training Test Dataset: ", sms_train_test)

# Training Dataset Shape
# logger.info("Training Dataset Shape: ", train_dataset.shape)
# logger.info("Test Dataset Shape: ", test_dataset.shape)
# logger.info("SSM Training Test Dataset Shape: ", sms_train_test.shape)

# Load the tokenizer for "distilbert-base-uncased" model.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
  # Pad/truncate each text to 512 tokens. Enforcing the same shape
  # could make the training faster.
  return tokenizer(
      examples["sms"],
      padding="max_length",
      truncation=True,
      max_length=128,
  )

seed = 22

# Tokenize the train and test datasets
train_tokenized = train_dataset.map(tokenize_function)
# logger.info("Tokenized Training Dataset Full: ", train_tokenized)
train_tokenized = train_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
# logger.info("Tokenized Training Dataset No SMS: ", train_tokenized)
# logger.info("Tokenized Training Dataset Data: ", train_tokenized[:5])


test_tokenized = test_dataset.map(tokenize_function)
# logger.info("Tokenized Test Dataset Full: ", test_tokenized)
test_tokenized = test_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
# logger.info("Tokenized Test Dataset No SMS: ", test_tokenized)

# Set the mapping between int label and its meaning.
id2label = {0: "ham", 1: "spam"}
label2id = {"ham": 0, "spam": 1}

# Acquire the model from the Hugging Face Hub, providing label and id mappings so that both we and the model can 'speak' the same language.
model = AutoModelForSequenceClassification.from_pretrained(
  "distilbert-base-uncased",
  num_labels=2,
  label2id=label2id,
  id2label=id2label,
)

# Define the target optimization metric
metric = evaluate.load("accuracy")


# Define a function for calculating our defined target optimization metric during training
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# Checkpoints will be output to this `training_output_dir`.
training_output_dir = "/tmp/sms_trainer"
training_args = TrainingArguments(
  output_dir=training_output_dir,
  evaluation_strategy="epoch",
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  logging_steps=8,
  num_train_epochs=3,
)

# Instantiate a `Trainer` instance that will be used to initiate a training run.
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_tokenized,
  eval_dataset=test_tokenized,
  compute_metrics=compute_metrics,
)

# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Pick a name that you like and reflects the nature of the runs that you will be recording to the experiment.
mlflow.set_experiment("Spam Classifier Training")

with mlflow.start_run() as run:
  trainer.train()


# If you're going to run this on something other than a Macbook Pro, change the device to the applicable type. "mps" is for Apple Silicon architecture in torch.
tuned_pipeline = pipeline(
  task="text-classification",
  model=trainer.model,
  batch_size=8,
  tokenizer=tokenizer,
  device="mps",
)

# Perform a validation of our assembled pipeline that contains our fine-tuned model.
quick_check = (
  "I have a question regarding the project development timeline and allocated resources; "
  "specifically, how certain are you that John and Ringo can work together on writing this next song? "
  "Do we need to get Paul involved here, or do you truly believe, as you said, 'nah, they got this'?"
)

logger.info("Pipeline Validation With Realistic Input: ", tuned_pipeline(quick_check))

# Define a set of parameters that we would like to be able to flexibly override at inference time, along with their default values
model_config = {"batch_size": 8}

# Infer the model signature, including a representative input, the expected output, and the parameters that we would like to be able to override at inference time.
signature = mlflow.models.infer_signature(
  ["This is a test!", "And this is also a test."],
  mlflow.transformers.generate_signature_output(
      tuned_pipeline, ["This is a test response!", "So is this."]
  ),
  params=model_config,
)

# Log the pipeline to the existing training run
with mlflow.start_run(run_id=run.info.run_id):
  model_info = mlflow.transformers.log_model(
      transformers_model=tuned_pipeline,
      artifact_path="fine_tuned",
      signature=signature,
      input_example=["Pass in a string", "And have it mark as spam or not."],
      model_config=model_config,
  )

# Load our saved model in the native transformers format
loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)

# Define a test example that we expect to be classified as spam
validation_text = (
  "Want to learn how to make MILLIONS with no effort? Click HERE now! See for yourself! Guaranteed to make you instantly rich! "
  "Don't miss out you could be a winner!"
)

# validate the performance of our fine-tuning
logger.info("Validation of the Loaded Fine-Tuned Model With Realistic Input: ", loaded(validation_text))