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
     extras = transform_extras(extras)

  combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  single_prompt = single_prompt + extras + "DOCUMENT: {}"
  summary, extracted_features = summarize(document, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
  aggregate_features(extracted_features)
  return summary, extracted_features

def transform_extras(extra_info):
  """
  inputs:
  extra_info: a comma seperated list of desired features: extra1, extra2,extra3,...
  outputs:
  the extras portion of the prompt
  """
  extras = extra_info.replace(" ", "").split(",")

  output = "Return the summary in JSON format, where \"summary\" is the key corresponding to the generated summary. \
    Also include the additional keys "

  for ix, extra in enumerate(extras):
    if ix == len(extras) - 1:
      output += "and \"" + extra + "\". "
    else:
      output += "\"" + extra + "\", "

  output += "Populate each of the additional keys (K) with a JSON object, where each key in that JSON object is \
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
    for instance in features[feature]:
      if len(features[feature][instance]) > 1:
        # aggregate
        prompt = f"Combine the following definitions/descriptions of \"{instance}\" into one cohesive definition/description: \
          {features[feature][instance]}. Make sure the output is in plain text."
        response = callGPT(prompt)
        features[feature][instance] = response



if __name__ == "__main__":

  with open("/Users/patricktimons/Documents/GitHub/document_summarizer/ds/summarization/thedead.txt", "r") as f:
    data = f.read()

  user_information = "kid"
  summary_details = "short"
  extras = "vocabulary, themes, characters"
  summary, features = prepare_summary(data, user_information,summary_details,extras)
  print("the summary: \n" + summary)
  print("the features: \n" + str(features))
  