import configparser
config = configparser.ConfigParser()

from aggregate_summary import aggregate_summary
from segment_input import segment_text

import tiktoken
import openai

import json

with open("config.json") as json_data_file:
  data = json.load(json_data_file)
  openai.api_key = data["summarization"]["OPENAI_API_KEY"]
  CONTEXT_WINDOW = int(data["summarization"]["CONTEXT_WINDOW"])


def summarize(input, CONTEXT_WINDOW=CONTEXT_WINDOW, summary_length=None):
    if summary_length == None:
      summary_legth = CONTEXT_WINDOW
    enc = tiktoken.encoding_for_model("gpt-4")
    MAX_SEGMENT_LENGTH = CONTEXT_WINDOW // 3 - 25 # calculated by fitting input texts, desired summary, and summarization directions into context window
    segmented_input = segment_text(input, SEGMENT_LENGTH=MAX_SEGMENT_LENGTH)
    summary = aggregate_summary(segmented_input, bandwidth=MAX_SEGMENT_LENGTH, output_length=summary_length)
    return summary


if __name__ == "__main__":
  with open("/Users/patricktimons/Documents/GitHub/document_summarizer/test.txt", "r") as f:
    data = f.read()
    summary = summarize(data, CONTEXT_WINDOW, CONTEXT_WINDOW*10)
    print(summary)