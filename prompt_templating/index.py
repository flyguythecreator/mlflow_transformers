from transformers import pipeline
import mlflow


generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

user_input = "Tell me the largest bird"
print("User Input: ", user_input)
prompt_templates = [
  # no template
  "{prompt}",
  # question-answer style template
  "Q: {prompt} A: ",
  # dialogue style template with a system prompt
  (
      "You are an assistant that is knowledgeable about birds. "
      "If asked about the largest bird, you will reply 'Duck'. "
      "User: {prompt} "
      "Assistant: "
  ),
]
print("Prompt Template: ", prompt_templates)
responses = generator(
  [template.format(prompt=user_input) for template in prompt_templates], max_new_tokens=15
)
for idx, response in enumerate(responses):
  print(f"Response to Template #{idx}:")
  print(response[0]["generated_text"] + "")

# Creating a Model Signature of MLFlow Monitoring 
sample_input = "Tell me the largest bird"
params = {"max_new_tokens": 15}
signature = mlflow.models.infer_signature(
  sample_input,
  mlflow.transformers.generate_signature_output(generator, sample_input, params=params),
  params=params,
)

# visualize the signature
print("Visualizing the Generated MLFlow Model Signature: ", signature)

# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set a name for the experiment that is indicative of what the runs being created within it are in regards to
mlflow.set_experiment("prompt-templating")

prompt_template = "Q: {prompt}/n A:"
with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
      transformers_model=generator,
      artifact_path="model",
      task="text-generation",
      signature=signature,
      input_example="Tell me the largest bird",
      prompt_template=prompt_template,
      # Since MLflow 2.11.0, you can save the model in 'reference-only' mode to reduce storage usage by not saving
      # the base model weights but only the reference to the HuggingFace model hub. To enable this, uncomment the
      # following line:
      save_pretrained=False,
  )

loaded_generator = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

print("Loaded Model Prediction: ", loaded_generator.predict("Tell me the largest bird"))