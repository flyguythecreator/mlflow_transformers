# Disable tokenizers warnings when constructing pipelines
# %env TOKENIZERS_PARALLELISM=false
import warnings
# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

import transformers
import mlflow

model_architecture = "google/flan-t5-base"

translation_pipeline = transformers.pipeline(
  task="translation_en_to_fr",
  model=transformers.T5ForConditionalGeneration.from_pretrained(
      model_architecture, max_length=1000
  ),
  tokenizer=transformers.T5TokenizerFast.from_pretrained(model_architecture, return_tensors="pt"),
)

# Evaluate the pipeline on a sample sentence prior to logging
print("Model Evaluation Test Prompt: ", translation_pipeline(
  "translate English to French: I enjoyed my slow saunter along the Champs-Élysées."
))

# Define the parameters that we are permitting to be used at inference time, along with their default values if not overridden
model_params = {"max_length": 1000}

# Generate the model signature by providing an input, the expected output, and (optionally), parameters available for overriding at inference time
signature = mlflow.models.infer_signature(
  "This is a sample input sentence.",
  mlflow.transformers.generate_signature_output(translation_pipeline, "This is another sample."),
  params=model_params,
)

# Visualize the model signature
print("Generated Model Signature: ", signature)

# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Translation")

with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
      transformers_model=translation_pipeline,
      artifact_path="english_to_french_translator",
      signature=signature,
      model_params=model_params,
  )
  print("Logged Model Info: ", model_info)

  # 
# Load our saved model as a dictionary of components, comprising the model itself, the tokenizer, and any other components that were saved
translation_components = mlflow.transformers.load_model(
  model_info.model_uri, return_type="components"
)

# Show the components that made up our pipeline that we saved and what type each are
for key, value in translation_components.items():
  print("Pipeline Components: ", f"{key} -> {type(value).__name__}")

# Show the model parameters that were saved with our model to gain an understanding of what is recorded when saving a transformers pipeline
print("Model Parameters: ", model_info.flavors)

# Load our saved model as a transformers pipeline and validate the performance for a simple translation task
translation_pipeline = mlflow.transformers.load_model(model_info.model_uri)
response = translation_pipeline("I have heard that Nice is nice this time of year.")

print("Translation Testing of Complex Sentences: ", response)

# Verify that the components that we loaded can be constructed into a pipeline manually
reconstructed_pipeline = transformers.pipeline(**translation_components)

reconstructed_response = reconstructed_pipeline(
  "transformers makes using Deep Learning models easy and fun!"
)

print("Verifying Loaded Model Integrity: ", reconstructed_response)

# View the components that were saved with our model
print("The Saved Language Translation Model Components: ", translation_components.keys())

# Access the individual components from the components dictionary
tokenizer = translation_components["tokenizer"]
model = translation_components["model"]

query = "Translate to French: Liberty, equality, fraternity, or death."

# This notebook was run on a Mac laptop, so we'll send the output tensor to the "mps" device.
# If you're running this on a different system, ensure that you're sending the tensor output to the appropriate device to ensure that
# the model is able to read it from memory.
inputs = tokenizer.encode(query, return_tensors="pt").to("mps")
outputs = model.generate(inputs).to("mps")
result = tokenizer.decode(outputs[0])

# Since we're not using a pipeline here, we need to modify the output slightly to get only the translated text.
print("Direct Output from Saved Language Translation Model: ", result.replace("<pad> ", "").replace("</s>", ""))

