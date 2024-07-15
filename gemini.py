import pathlib
import textwrap
import google.generativeai as genai
import os

#GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)


model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("What is the meaning of life?")
print(response)

