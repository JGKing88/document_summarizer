import tiktoken
import threading
import openai
import json

def aggregate_summary(input, combine_prompt, single_prompt, bandwidth, output_length, aux_attr=None):
  """
  input: list of summaries (each summary is a string)
  bandwidth: integer. What is our token bandwidth for the summaries. Also determines final summary length
  aggreagates summaries by merging first two summaries, then next two, and so on, then recursing
  assumptions:
  (1) each summary is less than half as many tokens as the context window
  """
  if aux_attr == None:
    aux_attr = dict()

  enc = tiktoken.encoding_for_model("gpt-4")

  # current length of output (only gets shorter as we iterate through input's indices)
  cur_agg_length = [sum(len(enc.encode(summary)) for summary in input)]
  output = [""]*(len(input)//2)
  if len(input)%2 == 1:
    output.append("")
  threads = []
  output_index = 0
  for i in range(0, len(input), 2):
    # merge
    if i+1 < len(input):
      # skip if combined with previous index
      if i%2 == 1:
        continue
      # merge case
      threads.append(threading.Thread(target=combine_summaries, args=(input[i], input[i+1], combine_prompt, bandwidth, output, output_index, cur_agg_length, aux_attr)))
      output_index += 1
    else:
      # lone case
      threads.append(threading.Thread(target=combine_summaries, args=(input[i], None, single_prompt, bandwidth, output, output_index, cur_agg_length, aux_attr, False)))
      output_index += 1

  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  # recursive caseL current output is too long:

  if cur_agg_length[0] > output_length:
    return aggregate_summary(output, combine_prompt, single_prompt, bandwidth, output_length, aux_attr)
  
  # get final output to desired length
  output = [str(i) for i in output]
  output_str = " ".join(output)
  return output_str, aux_attr



def combine_summaries(text1, text2, prompt, bandwidth, output, output_index, cur_agg_length, aux_attr, combine_summaries=True):
  print("combine_summaries called")
  enc = tiktoken.encoding_for_model("gpt-4")
  if combine_summaries:
    prompt = prompt.format(text1, text2)
  else:
    prompt = prompt.format(text1)
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  try:
    response_message = json.loads(response.choices[0].message.content)
  except:
    print("ERROR: unable to load GPT resonse as a json")
    return ""
  
  for attr in response_message:
    if attr == "summary":
      output[output_index] = response_message[attr]
      if combine_summaries:
        cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text1)) - len(enc.encode(text2))
      else:
        cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text1))

    else:
      if attr not in aux_attr:
        aux_attr[attr] = set()
      for item in response_message[attr]:
        aux_attr[attr].add(item)