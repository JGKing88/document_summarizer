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
  enc = tiktoken.encoding_for_model("gpt-4")

  # current length of output (only gets shorter as we iterate through input's indices)
  cur_agg_length = [sum(len(enc.encode(summary)) for summary in input)]
  output = [""]*len(input)
  messages = []
  threads = []
  output_index = 0
  for i in range(0, len(input), 1):
    # merge
    if i+1 < len(input) and (cur_agg_length[0] > output_length):
      # skip if combined with previous index
      if i%2 == 1:
        continue
      threads.append(threading.Thread(target=combine_summaries, args=(input[i], input[i+1], bandwidth, output, output_index, cur_agg_length)))
      output_index += 1

    # lone summarization: no merging
    else:
      threads.append(threading.Thread(target=lone_summary, args=(input[i], bandwidth, output, output_index, cur_agg_length)))
      output_index += 1


  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  

  # recursive case: current output is too long
  if cur_agg_length[0] > output_length:
    return aggregate_summary(output, bandwidth, output_length)
  
  # handles output is empty list
  if len(output) == 0:
    return ""
  
  # get final output to desired length
  output_str = " ".join(output)
  return output_str

def combine_summaries(text1, text2, bandwidth, output, output_index, cur_agg_length):
  enc = tiktoken.encoding_for_model("gpt-4")
  prompt = "Combine the following two summaries of adjacent text into one summary. SUMMARY 1: " + text1 + " SUMMARY 2: " + text2
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  output[output_index] = response.choices[0].message.content
  cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text1)) - len(enc.encode(text2))

def lone_summary(text, bandwidth, output, output_index, cur_agg_length):
  enc = tiktoken.encoding_for_model("gpt-4")
  prompt = "Summarize the following text" + text
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  output[output_index] = response.choices[0].message.content
  output_index += 1
  cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text))