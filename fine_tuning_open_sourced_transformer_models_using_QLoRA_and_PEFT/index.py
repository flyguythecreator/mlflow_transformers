import pandas as pd
from datasets import load_dataset
from IPython.display import HTML, display
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime
import mlflow
from mlflow.models import infer_signature


base_model_id = "mistralai/Mistral-7B-v0.1"
dataset_name = "b-mc2/sql-create-context"
dataset = load_dataset(dataset_name, split="train")

# Prepair the dataset
def display_table(dataset_or_sample):
  # A helper fuction to display a Transformer dataset or single sample contains multi-line string nicely
  pd.set_option("display.max_colwidth", None)
  pd.set_option("display.width", None)
  pd.set_option("display.max_rows", None)

  if isinstance(dataset_or_sample, dict):
      df = pd.DataFrame(dataset_or_sample, index=[0])
  else:
      df = pd.DataFrame(dataset_or_sample)

  html = df.to_html().replace("\n", "<br>")
  styled_html = f"""<style> .dataframe th, .dataframe tbody td {{ text-align: left; padding-right: 30px; }} </style> {html}"""
  display(HTML(styled_html))


print(display_table(dataset.select(range(3))))

# Split the dataset
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Training dataset contains {len(train_dataset)} text-to-SQL pairs")
print(f"Test dataset contains {len(test_dataset)} text-to-SQL pairs")
print("Traning Dataset: ", train_dataset)
print("Traning Dataset Shape: ", train_dataset.shape)
print("Testing Dataset: ", test_dataset)
print("Testing Dataset Shape: ", test_dataset.shape)

# Define the prompt template
PROMPT_TEMPLATE = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

### Table:
{context}

### Question:
{question}

### Response:
{output}"""


def apply_prompt_template(row):
  prompt = PROMPT_TEMPLATE.format(
      question=row["question"],
      context=row["context"],
      output=row["answer"],
  )
  return {"prompt": prompt}


train_dataset = train_dataset.map(apply_prompt_template)
print(display_table(train_dataset.select(range(1))))

# Pad the Training Dataset
## You can use a different max length if your custom dataset has shorter/longer input sequences.
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(
  base_model_id,
  model_max_length=MAX_LENGTH,
  padding_side="left",
  add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_and_pad_to_fixed_length(sample):
  result = tokenizer(
      sample["prompt"],
      truncation=True,
      max_length=MAX_LENGTH,
      padding="max_length",
  )
  result["labels"] = result["input_ids"].copy()
  return result


tokenized_train_dataset = train_dataset.map(tokenize_and_pad_to_fixed_length)

assert all(len(x["input_ids"]) == MAX_LENGTH for x in tokenized_train_dataset)

print(display_table(tokenized_train_dataset.select(range(1))))

# Load the Base Model (with 4-bit quantization)
quantization_config = BitsAndBytesConfig(
  # Load the model with 4-bit quantization
  load_in_4bit=True,
  # Use double quantization
  bnb_4bit_use_double_quant=True,
  # Use 4-bit Normal Float for storing the base model weights in GPU memory
  bnb_4bit_quant_type="nf4",
  # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
  bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=quantization_config)

## Test Vanilla Mistral Model
# tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# pipeline = transformers.pipeline(model=model, tokenizer=tokenizer, task="text-generation")

# sample = test_dataset[1]
# prompt = PROMPT_TEMPLATE.format(
#   context=sample["context"], question=sample["question"], output=""
# )  # Leave the answer part blank

# with torch.no_grad():
#   response = pipeline(prompt, max_new_tokens=256, repetition_penalty=1.15, return_full_text=False)

# display_table({"prompt": prompt, "generated_query": response[0]["generated_text"]})


# Define a PEFT Model
# Enabling gradient checkpointing, to make the training further efficient
model.gradient_checkpointing_enable()
# Set up the model for quantization-aware training e.g. casting layers, parameter freezing, etc.
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
  task_type="CAUSAL_LM",
  # This is the rank of the decomposed matrices A and B to be learned during fine-tuning. A smaller number will save more GPU memory but might result in worse performance.
  r=32,
  # This is the coefficient for the learned ΔW factor, so the larger number will typically result in a larger behavior change after fine-tuning.
  lora_alpha=64,
  # Drop out ratio for the layers in LoRA adaptors A and B.
  lora_dropout=0.1,
  # We fine-tune all linear layers in the model. It might sound a bit large, but the trainable adapter size is still only **1.16%** of the whole model.
  target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
      "lm_head",
  ],
  # Bias parameters to train. 'none' is recommended to keep the original model performing equally when turning off the adapter.
  bias="none",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
print(peft_model)

# Kickoff Training Job
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Comment-out this line if you are running the tutorial on Databricks
mlflow.set_experiment("MLflow PEFT Tutorial")

training_args = TrainingArguments(
  # Set this to mlflow for logging your training
  report_to="mlflow",
  # Name the MLflow run
  run_name=f"Mistral-7B-SQL-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
  # Replace with your output destination
  output_dir="YOUR_OUTPUT_DIR",
  # For the following arguments, refer to https://huggingface.co/docs/transformers/main_classes/trainer
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  gradient_checkpointing=True,
  optim="paged_adamw_8bit",
  bf16=True,
  learning_rate=2e-5,
  lr_scheduler_type="constant",
  max_steps=500,
  save_steps=100,
  logging_steps=100,
  warmup_steps=5,
  # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
  ddp_find_unused_parameters=False,
)

trainer = transformers.Trainer(
  model=peft_model,
  train_dataset=tokenized_train_dataset,
  data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
  args=training_args,
)

# use_cache=True is incompatible with gradient checkpointing.
peft_model.config.use_cache = False


with mlflow.start_run() as run:
  trainer.train()


# Infrense Prompt Template
# Basically the same format as we applied to the dataset. However, the template only accepts {prompt} variable so both table and question need to be fed in there.
prompt_template = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

{prompt}

### Response:
"""

# Infrence Parameters to Track
sample = train_dataset[1]

# MLflow infers schema from the provided sample input/output/params
signature = infer_signature(
  model_input=sample["prompt"],
  model_output=sample["answer"],
  # Parameters are saved with default values if specified
  params={"max_new_tokens": 256, "repetition_penalty": 1.15, "return_full_text": False},
)
print("Model Infrense Signature: ", signature)

# Get the ID of the MLflow Run that was automatically created above
last_run_id = mlflow.last_active_run().info.run_id

# Save a tokenizer without padding because it is only needed for training
tokenizer_no_pad = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)

# If you interrupt the training, uncomment the following line to stop the MLflow run
mlflow.end_run()

with mlflow.start_run(run_id=last_run_id):
  mlflow.log_params(peft_config.to_dict())
  mlflow.transformers.log_model(
      transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
      prompt_template=prompt_template,
      signature=signature,
      artifact_path="model",  # This is a relative path to save model files within MLflow run
      # Uncomment the following line to save the model in 'reference-only' mode:
      save_pretrained=False,
  )


# We only input table and question, since system prompt is adeed in the prompt template.
test_prompt = """
### Table:
CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)

### Question:
When Essendon played away; where did they play?
"""
mlflow.get_run()
# You can find the ID of run in the Run detail page on MLflow UI
mlflow_model = mlflow.pyfunc.load_model(f"runs:/{last_run_id}/model")

# Inference parameters like max_tokens_length are set to default values specified in the Model Signature
generated_query = mlflow_model.predict(test_prompt)[0]
print(display_table({"prompt": test_prompt, "generated_query": generated_query}))