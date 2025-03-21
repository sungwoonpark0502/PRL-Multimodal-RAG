import google.generativeai as genai
genai.configure(api_key="AIzaSyDyC5jmGivHO15aVKN2jEim5SmaKLu-xiA")

model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
response = model.generate_content("Who is Cristiano Ronaldo?")
print(response.text)
