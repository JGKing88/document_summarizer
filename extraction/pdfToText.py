import pdftotext
from transformers import GPTJForCausalLM, AutoTokenizer
import torch 

# Load your PDF
with open("example.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)
# print(len(pdf)) #number of pages
# for page in pdf:
#    print(page)
#
s = "\n\n".join(pdf) #join all strings
split = s.split('\n')
corrected_string = []
for line in split:
    if len(line.split(' '))>1:
        corrected_string.append(line)
final_text = "\n".join(corrected_string)
#print(final_text)
#print(s)   
print("Enter any relevant information you think is important for the model to know in order to summarize your text (age, demographics, English reading experience/levels):")
demographic = input()
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
context = """You will be given an article and some demographic information about a reader. Summarize the article in a way that a user of this demographic finds easy to understand. Demographics: """ + demographic + """ \n Article: """+ final_text + """ Now generate the article summary: """

input_ids = tokenizer(context, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

