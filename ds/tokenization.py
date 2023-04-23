import tiktoken
import math

ENC = tiktoken.encoding_for_model("gpt-4")

def tok_doc(doc_string):
    """
    Tokenizes the document and returns the encoding.
    """
    return ENC.encode(doc_string)

def tok_len(enc_output):
    """
    Returns the number of tokens in the document.
    """
    return len(enc_output)

def max_sum_len(enc_output):
    """
    Returns the maximum  of tokens in the document.
    """
    return math.floor(0.2*tok_len(enc_output))
