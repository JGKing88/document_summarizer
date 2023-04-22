import argparse
import configparser

from .summarization.summaraize import summarize

parser = argparse.ArgumentParser(
                    prog='Document Summarizer',
                    description='',
                    epilog='Text at the bottom of help')

parser.add_argument('filename')           # positional argument
parser.add_argument('-k', '--api_key')      # option that takes a value
parser.add_argument('-w', '--cotext_window')
parser.add_argument('-l', '--length')  # desired length of the summary



args = parser.parse_args()

config = configparser.ConfigParser()
config['DEFAULT'] = {'OPENAI_API_KEY': args.api_key,
                     'CONTEXT_WINDOW': None,
                     'SUMMARY_LENGTH': None}

if args.context_window is not None:
  config['DEFAULT']['CONTEXT_WINDOW'] = args.context_window

if args.summary_length is not None:
    config['DEFAULT']['SUMMARY_LENGTH'] = int(args.length)


with open('example.ini', 'w') as configfile:
  config.write(configfile)

with open(config['DEFAULT']['filename']) as input_file:
    raw_text = pdftotext(input_file)
    print(summarize(raw_text))