import openai
openai.api_key = "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"

def callGPT(prompt_prompt, max_sum_len=1000):
  prompt = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
              {"role": "user", "content": prompt_prompt},
          ],
      temperature=0.2, #[0,2] higher values are more random (want low randomnes)
      max_tokens=max(1000, max_sum_len),
  )
  return prompt.choices[0].message.content

doc_title = "'Attention is All you Need'"
user_info = "A PhD candidate at MIT who is studying A.I."


prelim_prompt =f"Generate a prompt to tell ChatGPT to define each of the following features \
    extracted from {doc_title}. Make sure that all definitions are appropriate for this type of user: \
      {user_info}. Also, make sure that all definitions are relative to the context of {doc_title}: \
        {feature_class}: {features[feature_class]}"
prompt = callGPT(prelim_prompt)
print(prompt)