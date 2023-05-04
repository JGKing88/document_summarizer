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
  
  if "document title" in features:
    doc_title = features["document title"]
  else:
    doc_title = "a text document"
  
  
  output = dict()
  for feature_class in features:
    if feature_class == "document title":
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
     extras = transform_extras(extras)
     print("extras: " + extras)

  combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  single_prompt = single_prompt + extras + "DOCUMENT: {}"
  summary, extracted_features = summarize(document, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
  print('extracted features :\n' + str(extracted_features))
  return summary

def transform_extras(extra_info):
  """
  will generate the extras portion of the prompt
  assumes that extra_info is seperated by commas: extra1, extra2,extra3,...
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



if __name__ == "__main__":

  with open("/Users/patricktimons/Documents/GitHub/document_summarizer/ds/summarization/thedead.txt", "r") as f:
    data = f.read()

  user_information = "kid"
  summary_details = "short"
  extras = "vocabulary, themes, characters"
  summary = prepare_summary(data, user_information,summary_details,extras)
  print("the summary: \n" + summary)
  
  # single_prompt_prompt = "Generate a prompt to tell ChatGPT to generate a summary of a document \
  #   for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  # single_prompt = callGPT(single_prompt_prompt)

  # combine_prompt_prompt = "Generate a prompt to tell ChatGPT to combine two summaries into one summary \
  #   for this type of user: " + user_information + " and this type of summary: " + summary_details + ". "
  # combine_prompt = callGPT(combine_prompt_prompt)

  # if extras != "":
  #    extras = " Along with the summary, include " + extras + ". Make sure the response is in JSON format. The key of the summary should be 'summary', \
  #     the key of the title should be 'document title', and so on."

  # print(combine_prompt + extras)
  # print(single_prompt + extras)
  
  # combine_prompt = combine_prompt + extras + "SUMMARY 1: {} SUMMARY 2: {}"
  # single_prompt = single_prompt + extras + "DOCUMENT: {}"
  # summary, features = summarize(data, combine_prompt, single_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
  # print("summary", summary)
  # print("extra info", features)
  # features = {'document title': 'A Festive Gathering', 'characters': {'Miss Julia', 'Mary Jane', 'Freddy Malins', "Mr. Bartell D'Arcy", 'Michael Furey', 'Mrs Malins', "Mr Bartell D'Arcy", 'Mr Browne', 'Gretta', 'Miss Furlong', 'Mr. Browne', 'Aunt Julia', 'Miss Kate', 'Miss Ivors', 'Aunt Kate', 'Miss Daly', 'Gabriel', 'Lily', 'Mrs. Conroy', "Miss O'Callaghan"}, 'themes': {'Irish hospitality', 'Marriage', 'Laughter', 'Tradition', 'Festive Gatherings', 'Food and Drink', 'Disagreements', 'Music and Opera', 'Celebration', 'Nostalgia', 'Passion', 'Death', 'Memory', 'Resentment', 'Cultural heritage', 'Family', 'Dance', 'Love', 'Music', 'Regret'}}
  # print("definitons: \n", define_features(features, user_information))
  # print(callGPT("how many keys are in this json object: " + str(features)))
  # print(transform_extras(extras))