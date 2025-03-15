import google.generativeai as genai
import config  # Ensure API key is set

models = genai.list_models()
for model in models:
    print(model)
