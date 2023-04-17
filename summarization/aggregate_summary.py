import tiktoken
import threading
import openai

def aggregate_summary(input, bandwidth, output_length):
  """
  input: list of summaries (each summary is a string)
  bandwidth: integer. What is our token bandwidth for the summaries. Also determines final summary length
  aggreagates summaries by merging first two summaries, then next two, and so on, then recursing
  assumptions:
  (1) each summary is less than half as many tokens as the context window
  """
  threads = []
  output = []
  enc = enc = tiktoken.encoding_for_model("gpt-4")
  for i in range(0, len(input), 2):
    if i+1 < len(input):
      threads.append(threading.Thread(target=aggregate_pair, args=(input[i], input[i+1], bandwidth, output)))
    else:
      threads.append(threading.Thread(target=lone_text, args=(input[i], output)))
  for thread in threads:
    print("starting thread", thread.ident)
    thread.start()
  for thread in threads:
    print("joining thread", thread.ident)
    thread.join()
  if len(output) > 1:
    return aggregate_summary(output, bandwidth, output_length)
  
  if len(output) == 0:
    return ""
  
  # get final output to desired length
  if len(enc.encode(output[0])) <= output_length:
    return output[0]
  prompt = "Summarize the following text: " + output[0]
  response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=output_length,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
      )

  return response["choices"][0]["text"]

def aggregate_pair(text1, text2, bandwidth, output):
  prompt = "Combine the following two summaries of adjacent text into one summary. SUMMARY 1: " + text1 + " SUMMARY 2: " + text2
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=bandwidth,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  output.append(response["choices"][0]["text"])

def lone_text(text, output):
  output.append(text)