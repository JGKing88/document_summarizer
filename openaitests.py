import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a summarizer. Summarize the next document with the following context: This summary is for a blind person."},
        {"role": "user", "content": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus."}
    ],
  temperature=0.7,
  max_tokens=64,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# prompt_prompt = "Generate a prompt that tells ChatGPT to generate a summary of the following document for this type of user:  and this type of summary: " 
# prompt = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#             {"role": "user", "content": prompt_prompt},
#         ],
#     temperature=0.2, #[0,2] higher values are more random (want low randomnes)
#     max_tokens=64,
# )

print(response.choices[0].message.content)

