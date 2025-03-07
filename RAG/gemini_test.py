# Updated backend.py snippet
import google.generativeai as genai

genai.configure(
    api_key="GEMINI_KEY",
    transport="rest",
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

# Use the latest available model
model = genai.GenerativeModel('gemini-1.5-pro-latest')  # ✅ From your list

# Test
response = model.generate_content("What is 2+2?")
print(response.text)  # Should output "4"


# import google.generativeai as genai

# genai.configure(api_key="AIzaSyAMnsRENk5CRRN5osnwaH_B2tgz8VCrRGA")
# print("Available Models:")
# for model in genai.list_models():
#     print(f"- {model.name}")  # Print the model names