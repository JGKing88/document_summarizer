import tiktoken

def segment_text(input, SEGMENT_LENGTH=None, TOKEN_LENGTH=None):
  """
  breaks text into SEGMENT_LENGTH chunks
  will return an iterable [chunk_1, chunk_2, ..., chunk_n] 
  """
  enc = tiktoken.encoding_for_model("gpt-4")
  tokenized = enc.encode(input)

  output = []
  for i in range(0, len(tokenized), SEGMENT_LENGTH):
    if i + SEGMENT_LENGTH > len(tokenized):
      output.append(enc.decode(tokenized[i:]))
      print("length", len(tokenized[i:]))
    else:
        output.append(enc.decode(tokenized[i:i+SEGMENT_LENGTH]))
        print("length", len(tokenized[i:i+SEGMENT_LENGTH]))
    
  
  return output