import tiktoken
import threading
import openai

def aggregate_summary(input, combine_prompt, single_prompt, bandwidth, output_length):
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
  for i in range(0, len(input), 2):
    # merge
    if i+1 < len(input):
      # skip if combined with previous index
      if i%2 == 1:
        continue
      threads.append(threading.Thread(target=combine_summaries, args=(input[i], input[i+1], combine_prompt, bandwidth, output, output_index, cur_agg_length)))
      output_index += 1

  print(threads)

  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  
  # recursive case: current output is too long
  if len([x for x in output if x != ""]) > 1:
    return aggregate_summary(output, combine_prompt, single_prompt, bandwidth, output_length)
  
  # handles output is empty list
  if len(output) == 0:
    return ""
  
  # get final output to desired length
  output_str = " ".join(output)
  return output_str

def combine_summaries(text1, text2, prompt, bandwidth, output, output_index, cur_agg_length):
  enc = tiktoken.encoding_for_model("gpt-4")
  prompt = prompt.format(text1, text2)
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  output[output_index] = response.choices[0].message.content
  cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text1)) - len(enc.encode(text2))

def lone_summary(text, prompt, bandwidth, output, output_index, cur_agg_length):
  enc = tiktoken.encoding_for_model("gpt-4")
  prompt = prompt.format(text)
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  output[output_index] = response.choices[0].message.content
  cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text))