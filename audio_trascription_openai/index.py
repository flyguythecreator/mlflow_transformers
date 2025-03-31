import warnings
import requests
import transformers
import mlflow
import numpy as np
from sklearn import datasets, metrics

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

# Acquire an audio file that is in the public domain
resp = requests.get(
  "https://www.nasa.gov/wp-content/uploads/2015/01/590325main_ringtone_kennedy_WeChoose.mp3"
)
resp.raise_for_status()
audio = resp.content

# Set the task that our pipeline implementation will be using
task = "automatic-speech-recognition"

# Define the model instance
architecture = "openai/whisper-large-v3"

# Load the components and necessary configuration for Whisper ASR from the Hugging Face Hub
model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

# Instantiate our pipeline for ASR using the Whisper model
audio_transcription_pipeline = transformers.pipeline(
  task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)

# For Demo Purposes only, remove for production
def format_transcription(transcription):
  """
  Function for formatting a long string by splitting into sentences and adding newlines.
  """
  # Split the transcription into sentences, ensuring we don't split on abbreviations or initials
  sentences = [
      sentence.strip() + ("." if not sentence.endswith(".") else "")
      for sentence in transcription.split(". ")
      if sentence
  ]

  # Join the sentences with a newline character
  return "".join(sentences)

# Specify parameters and their defaults that we would like to be exposed for manipulation during inference time
model_config = {
  "chunk_length_s": 20,
  "stride_length_s": [5, 3],
}

# Define the model signature by using the input and output of our pipeline, as well as specifying our inference parameters that will allow for those parameters to
# be overridden at inference time.
signature = mlflow.models.infer_signature(
  audio,
  mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
  params=model_config,
)

# Visualize the signature
print(signature)

# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Whisper Transcription ASR")

# MLFlow Auto Logging
mlflow.autolog()

# Log the pipeline
with mlflow.start_run(run_name="Audio Transcription"):
  # Log the model to the existing training run
  model_info = mlflow.transformers.log_model(
      transformers_model=audio_transcription_pipeline,
      artifact_path="whisper_transcriber",
      signature=signature,
      input_example=audio,
      model_config=model_config,
      # Since MLflow 2.11.0, you can save the model in 'reference-only' mode to reduce storage usage by not saving
      # the base model weights but only the reference to the HuggingFace model hub. To enable this, uncomment the
      # following line:
      save_pretrained=False,
  )

# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

# Perform transcription with the native pipeline implementation
transcription = loaded_transcriber(audio)

# print(f"Whisper native output transcription: {format_transcription(transcription['text'])}")

# Load the saved transcription pipeline as a generic python function
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

# Ensure that the pyfunc wrapper is capable of transcribing passed-in audio
pyfunc_transcription = pyfunc_transcriber.predict([audio])

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.
# print(f"Pyfunc output transcription: {format_transcription(pyfunc_transcription[0])}")