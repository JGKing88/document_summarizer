import configparser
config = configparser.ConfigParser()

from aggregate_summary import aggregate_summary
from segment_input import segment_text

import openai
openai.api_key = config["DEFAULT"]["OPENAI_API_KEY"]
TOKEN_LIMIT = config["DEFAULT"]["CONTEXT_WINDOW"]
TOKEN_LENGTH = config["DEFAULT"]["TOKEN_LENGTH"]

def summarize(input, TOKEN_LIMIT=TOKEN_LIMIT, summary_length=None):
    MAX_SEGMENT_LENGTH = TOKEN_LIMIT // 2 -25 # calculated by fitting input text, desired summary, and summarization directions into context window
    if summary_length == None:
        summary_length = min(len(input)//4, MAX_SEGMENT_LENGTH)
    segmented_input = segment_text(input, SEGMENT_LENGTH=MAX_SEGMENT_LENGTH, TOKEN_LENGTH=TOKEN_LENGTH)
    summary = aggregate_summary(segmented_input, MAX_SEGMENT_LENGTH, summary_length)
    return summary