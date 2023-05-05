import openai
# import ds_config
import json
import logging

with open("config.json") as json_data_file:
  data = json.load(json_data_file)
  CONTEXT_WINDOW = int(data["summarization"]["CONTEXT_WINDOW"])
  openai.api_key = data["summarization"]["API_KEY"] #ds_config.api_key()

def callGPT(prompt_prompt, max_sum_len=1000):
  try:
    prompt = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "user", "content": prompt_prompt},
            ],
        temperature=0.2, #[0,2] higher values are more random (want low randomnes)
        max_tokens=max(1000, max_sum_len),
    )
    return prompt.choices[0].message.content
  except Exception as e:
    logging.warning("GPT call unsucessful")
    logging.warning(str(e.message), str(e.args))
    return ""