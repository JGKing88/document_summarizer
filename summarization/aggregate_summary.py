import openai
openai.api_key = "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"

import tiktoken

def aggregate_summary(input, TOKEN_LIMIT=None, SUMMARY_LENGTH=None):
  """
  input: list of summaries (each summary is a string)
  TOKEN_LIMIT: integer. What is our token bandwidth for the summaries. Also determines final summary length
  aggreagates summaries by merging first two summaries, then next two, and so on, then recursing
  assumptions:
  (1) each summary is less than half as many tokens as the context window
  """
  output = []
  enc = enc = tiktoken.encoding_for_model("gpt-4")
  for i in range(0, len(input), 2):
    print('i', i)
    if i+1 < len(input):
      # combine input[i] and input[i+1]
      prompt = "Combine the following two summaries of adjacent text into one summary. SUMMARY 1: " + input[i] + " SUMMARY 2: " + input[i+1]
      # print("tokens:", len(enc.encode(prompt)))
      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=TOKEN_LIMIT,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
      )
      combined = response["choices"][0]["text"]
      output.append(combined)
    else:
      output.append(input[i])
  if len(output) > 1:
    return aggregate_summary(output, TOKEN_LIMIT)
  
  if len(output) == 0:
    return ""
  
  # get final output to desired length
  prompt = "Summarize the following text: " + output[0]
  response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=SUMMARY_LENGTH,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
      )

  return response["choices"][0]["text"]