import configparser
config = configparser.ConfigParser()

from aggregate_summary import aggregate_summary
from segment_input import segment_text
import tiktoken
import openai

import json

openai.api_key = "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"
CONTEXT_WINDOW = 8000


def summarize(input, CONTEXT_WINDOW=CONTEXT_WINDOW, summary_length=None):
    if summary_length == None:
      summary_length = CONTEXT_WINDOW
    enc = tiktoken.encoding_for_model("gpt-4")
    MAX_SEGMENT_LENGTH = CONTEXT_WINDOW // 3 - 25 # calculated by fitting input texts, desired summary, and summarization directions into context window
    segmented_input = segment_text(input, SEGMENT_LENGTH=MAX_SEGMENT_LENGTH)
    summary = aggregate_summary(segmented_input, bandwidth=MAX_SEGMENT_LENGTH, output_length=summary_length)
    return summary


if __name__ == "__main__":
  with open("araby.txt", "r") as f:
    data = f.read()
    summary = summarize(data, CONTEXT_WINDOW, CONTEXT_WINDOW)
    print(summary)