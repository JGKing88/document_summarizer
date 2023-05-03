import configparser
config = configparser.ConfigParser()

from summarization.aggregate_summary import aggregate_summary
from summarization.segment_input import segment_text
import tiktoken
import openai
import ds_config
import json

with open("config.json") as json_data_file:
  data = json.load(json_data_file)
  CONTEXT_WINDOW = int(data["summarization"]["CONTEXT_WINDOW"])

openai.api_key = ds_config.api_key()

def summarize(input, combine_prompt, single_prompt, CONTEXT_WINDOW=CONTEXT_WINDOW, summary_length=None):
    if summary_length == None:
      summary_length = CONTEXT_WINDOW // 3 - len(enc.encode(combine_prompt))
    enc = tiktoken.encoding_for_model("gpt-4")
    MAX_SEGMENT_LENGTH = CONTEXT_WINDOW // 3 - len(enc.encode(combine_prompt)) # calculated by fitting input texts, desired summary, and summarization directions into context window
    segmented_input = segment_text(input, SEGMENT_LENGTH=MAX_SEGMENT_LENGTH)
    summary, aux_attr = aggregate_summary(segmented_input, combine_prompt, single_prompt, bandwidth=MAX_SEGMENT_LENGTH, output_length=summary_length)
    return summary, aux_attr

def define_features(features, user_info=""):
  """
  input: a dict of featueres {'feature_class': {feature1, feature2, ...}}
  output: a dict of features to definitions
  """
  enc = tiktoken.encoding_for_model("gpt-4")
  if user_info == "":
    user_info = "an average user of a document summarization service."
  
  if "document_title" in features:
    doc_title = features["document_title"]
  else:
    doc_title = "a text document"
  
  
  output = dict()
  for feature_class in features:
    if feature_class == "document_title":
      continue

    prompt = f"Define each of the following features extracted from {doc_title}. Make sure that all \
      definitions are appropriate for this type of user: {user_info}. Also make sure that all \
        definitions are relative to the context of {doc_title}. {feature_class}: {features}"
    max_len = 3000 # CONTEXT_WINDOW - sum(len(enc.encode(feature)) for feature in features[feature_class]) - len(enc.encode(prompt))
    response = callGPT(prompt, max_len)
    output[feature_class] = response
  return output

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

def prepare_summary(document, user_information="", summary_details="", extras=""):
  single_prompt_prompt = "Generate a prompt to tell ChatGPT to generate a summary of a document \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  single_prompt = callGPT(single_prompt_prompt)

  combine_prompt_prompt = "Generate a prompt to tell ChatGPT to combine two summaries into one summary \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  combine_prompt = callGPT(combine_prompt_prompt)

  if extras != "":
     extras = " Along with the summary, include " + extras + ". Make sure the response is in JSON format. The key of the summary should be 'summary'."

  combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  single_prompt = single_prompt + extras + "DOCUMENT: {}"
  summary = summarize(document, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)

  return summary

if __name__ == "__main__":

  with open("/Users/patricktimons/Documents/GitHub/document_summarizer/ds/summarization/thedead.txt", "r") as f:
    data = f.read()

  user_information = "kid"
  summary_details = "short"
  extras = "a list of characters and a list of themes"
  
  single_prompt_prompt = "Generate a prompt to tell ChatGPT to generate a summary of a document \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  single_prompt = callGPT(single_prompt_prompt)

  combine_prompt_prompt = "Generate a prompt to tell ChatGPT to combine two summaries into one summary \
    for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  combine_prompt = callGPT(combine_prompt_prompt)

  if extras != "":
     extras = " Along with the summary, include " + extras + ". Make sure the response is in JSON format. The key of the summary should be 'summary'."

  print(combine_prompt + extras)
  print(single_prompt + extras)
  
  combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  single_prompt = single_prompt + extras + "DOCUMENT: {}"
  summary, features = summarize(data, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
  print("summary", summary[0])
  print("extra info", summary[1])
  print("definitons: \n", define_features(features, user_information))