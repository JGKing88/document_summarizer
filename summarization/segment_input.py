def segment_text(input, SEGMENT_LENGTH=(8000-15)/2):
  """
  breaks text into SEGMENT_LENGTH chunks
  will return an iterable [chunk_1, chunk_2, ..., chunk_n] 
  """
  output = []
  # assumption 1: a token is always 4 characters
  cur_chunk = ""
  ix = 0
  while ix < len(input):
    if len(cur_chunk) < SEGMENT_LENGTH * 4: # assumption 1
      cur_chunk += input[ix]
    else:
      output.append(cur_chunk)
      cur_chunk = ""

    ix+=1
  return output

segment_text("Hello world hello world hello world hello world hello world hello world ", 3)

