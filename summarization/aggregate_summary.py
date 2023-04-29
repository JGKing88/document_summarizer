import tiktoken
import threading
import openai
import json

from segment_input import segment_text
from TODO_JACK_MAKE_THIS import generate_prompt

def aggregate_summary(input, bandwidth, output_length, custom_prompt = None, user_info=None, aux_attr = None):
  """
  input: list of summaries (each summary is a string)
  bandwidth: integer. What is our token bandwidth for the summaries. Also determines final summary length
  aggreagates summaries by merging first two summaries, then next two, and so on, then recursing
  assumptions:
  (1) each summary is less than half as many tokens as the context window
  """
  if aux_attr == None:
    aux_attr = dict()

  if user_info == None:
    user_info = "Average user of this summarizer"

  if custom_prompt == None:
    custom_prompt = generate_prompt(user_info)

  enc = tiktoken.encoding_for_model("gpt-4")

  # current length of output (only gets shorter as we iterate through input's indices)
  cur_agg_length = [sum(len(enc.encode(summary)) for summary in input)]
  output = [""]*len(input)
  threads = []
  output_index = 0
  for i in range(0, len(input), 1):
    threads.append(threading.Thread(target=summarize_chunk, args=(input[i], bandwidth, output, output_index, cur_agg_length, aux_attr)))
    output_index += 1

  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  
  # edge case: output is empty
  if len(output) == 0:
    return ""

  # recursive case: current output is too long
  output_str = " ".join(output)
  output = segment_text(output_str, SEGMENT_LENGTH=bandwidth*3)
  if cur_agg_length[0] > output_length:
    return aggregate_summary(output, bandwidth, output_length, custom_prompt=custom_prompt, user_info=user_info, aux_attr=aux_attr)
  
  # output_str is correct length
  return output_str, aux_attr

def summarize_chunk(text, bandwidth, output, output_index, cur_agg_length, custom_prompt, aux_attr):
  enc = tiktoken.encoding_for_model("gpt-4")
  prompt = custom_prompt.format(text) # may need to change
  message = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=bandwidth,
    )
  # assume response is a string in format "{'summary': "the summary", 'attribute1': "the attribute"}"
  gpt_response = json.loads(response.choices[0].message.content)
  for attribute in gpt_response:
    if attribute == 'summary':
      output[output_index] = gpt_response[attribute]
      cur_agg_length[0] += len(enc.encode(output[output_index])) - len(enc.encode(text))
    else:
      # auxillary attribute such as vocabulary or concept (specific to user)
      if attribute in aux_attr:
        aux_attr[attribute].append(gpt_response[attribute])
      else:
        aux_attr[attribute] = [gpt_response[attribute]]