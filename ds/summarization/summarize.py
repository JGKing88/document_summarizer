import configparser
config = configparser.ConfigParser()

from aggregate_summary import aggregate_summary # from summarization.aggregate_summary import aggregate_summary
from segment_input import segment_text # from summarization.segment_input import segment_text
import tiktoken
import openai
# import ds_config
import json

with open("config.json") as json_data_file:
  data = json.load(json_data_file)
  CONTEXT_WINDOW = int(data["summarization"]["CONTEXT_WINDOW"])
  openai.api_key = data["summarization"]["API_KEY"] #ds_config.api_key()

from GPT import callGPT


def summarize(input, combine_prompt, single_prompt, CONTEXT_WINDOW=CONTEXT_WINDOW, summary_length=None):
    if summary_length == None:
      summary_length = CONTEXT_WINDOW // 3 - len(enc.encode(combine_prompt))
    enc = tiktoken.encoding_for_model("gpt-4")
    MAX_SEGMENT_LENGTH = CONTEXT_WINDOW // 3 - len(enc.encode(combine_prompt)) # calculated by fitting input texts, desired summary, and summarization directions into context window
    segmented_input = segment_text(input, SEGMENT_LENGTH=MAX_SEGMENT_LENGTH)
    summary, aux_attr = aggregate_summary(segmented_input, combine_prompt, single_prompt, bandwidth=MAX_SEGMENT_LENGTH, output_length=summary_length)
    return summary, aux_attr

def prepare_summary(document, user_information="", summary_details="", extras=""):
  single_prompt_prompt = "Generate a prompt to tell ChatGPT to generate a summary of a document \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  single_prompt = callGPT(single_prompt_prompt)

  combine_prompt_prompt = "Generate a prompt to tell ChatGPT to combine two summaries into one summary \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  combine_prompt = callGPT(combine_prompt_prompt)

  if extras != "":
     extras = transform_extras(extras)

  combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  single_prompt = single_prompt + extras + "DOCUMENT: {}"
  summary, extracted_features = summarize(document, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
  aggregate_features(extracted_features)
  if "title" in extracted_features:
    title = extracted_features["title"]
  else:
    title = "a text document"
  final_summary = final_summary_aggregation(summary, summary_details, title)
  return final_summary, extracted_features

def transform_extras(extra_info):
  """
  inputs:
  extra_info: a comma seperated list of desired features: extra1,extra2,extra3,...
  outputs:
  the extras portion of the prompt
  """
  extras = extra_info.split(",")

  output = "Return the summary in JSON format, where \"summary\" is the key corresponding to the generated summary, \
    Also include the additional keys: "

  for ix, extra in enumerate(extras):
    if ix == len(extras) - 1:
      output += "and \"" + extra + "\""
    else:
      output += "\"" + extra + "\", "

  output += ". Populate each of the additional keys (K) with a JSON object, where each key in that JSON object is \
    an instance of K in the text we are summarizing. For example, if our additional keys are 'vocabulary' and 'characters', \
      return: "
  example = {
    "summary": "example summary",
    "vocabulary": {
      "example word 1": "example definition 1",
      "example word 2": "example definition 2",
    },
    "characters": {
      "example character 1": "description of example character 1"
    }
  }
  output += str(example).replace('{', '{{').replace('}', '}}') + ". "
  return output

def aggregate_features(features):
  """
  input
  features: dictionary {"vocabulary": {"word": {"def1", "def2"}}}
  output: dictionary, but each instance of a feature (word/character) only has one defintion
  """
  for feature in features:
    if feature == "title":
      continue
    for instance in features[feature]:
      if len(features[feature][instance]) > 1:
        # aggregate
        prompt = f"Combine the following definitions/descriptions of \"{instance}\" into one cohesive definition/description: \
          {features[feature][instance]}. Make sure the output is in plain text."
        response = callGPT(prompt)
        features[feature][instance] = response
      else:
        features[feature][instance] = features[feature][instance].pop()

def final_summary_aggregation(summary, user_info="average reader", summary_details="average summary", title="a text document"):
  enc = tiktoken.encoding_for_model("gpt-4")
  prelim_prompt = f"The following text is an excerpt from concatenated summaries of different sections of {title}. \
      Rewrite these disjoint summaries into one cohesive summary by removing repeated information and adding \
        transitions as necessary. The final summary should have the following specifications: {summary_details}. \
          Additionally, the summary should be appropriate for the following type of reader: {user_info}. \
            If there is any text at the end that appears truncated, then after the summary, write \"[EXTRA TEXT]\", followed \
            by the text that gets truncated (along with a few sentences contextualizing the truncated text). "
  prompt_length = len(enc.encode(prelim_prompt))
  
  output_summary = ""
  cur_idx = 0
  trunc_text = ""
  while cur_idx < len(summary):
    summary_chunk = trunc_text + enc.decode(enc.encode(summary[cur_idx:])[:CONTEXT_WINDOW//2 - prompt_length - len(enc.encode(trunc_text))])
    prompt = prelim_prompt + summary_chunk
    response = callGPT(prompt, CONTEXT_WINDOW//2)
    response = response.split("[EXTRA TEXT]", maxsplit=1)
    output_summary += response[0]
    if len(response) > 1:
      trunc_text = response[1]
    else:
      trunc_text = ""
    cur_idx += CONTEXT_WINDOW//2 - prompt_length - len(enc.encode(trunc_text))
  
  return output_summary
    



if __name__ == "__main__":

  with open("/Users/patricktimons/Documents/GitHub/document_summarizer/test.txt", "r") as f:
    data = f.read()

  user_information = "10 year old with ADHD that rarely reads"
  summary_details = "2-3 paragraphs"
  extras = "vocabulary"
  summary, features = prepare_summary(data, user_information,summary_details,extras)
  print("the summary: \n" + summary)
  print("the features: \n" + str(features))
