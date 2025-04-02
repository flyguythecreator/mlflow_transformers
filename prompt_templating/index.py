from transformers import pipeline

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
      "If asked about the largest bird, you will reply 'Duck'."
      "User: {prompt}"
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