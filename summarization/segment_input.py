def segment_text(input, SEGMENT_LENGTH=None, TOKEN_LENGTH=4):
  """
  breaks text into SEGMENT_LENGTH chunks
  will return an iterable [chunk_1, chunk_2, ..., chunk_n] 
  assumptions: tokens are always 4 characters
  """
  if SEGMENT_LENGTH == None:
    raise AssertionError("Need to provide SEGMENT_LENGTH")
  
  output = []
  
  cur_chunk = ""
  ix = 0
  while ix < len(input):
    if len(cur_chunk) < SEGMENT_LENGTH * token_length: # assumption 1: a token is always 4 characters
      cur_chunk += input[ix]
    else:
      output.append(cur_chunk)
      cur_chunk = ""

    ix+=1
  return output

segment_text("Hello world hello world hello world hello world hello world hello world ", 3)

