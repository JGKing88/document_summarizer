from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import PyPDF2
from io import BytesIO
# import pdftotext
from transformers import GPTJForCausalLM, AutoTokenizer
import torch 


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    uploaded_file = request.files['pdf_file']
    user_information = request.form['user_information']
    
    # Process the PDF and user_information to generate a summary
    summary = process_pdf_and_user_information(uploaded_file, user_information)

    return jsonify({'summary': summary})

def process_pdf_and_user_information(uploaded_file, user_information):
    # Placeholder function for processing the PDF and user_information
    # You can implement your own summary generation logic here
    # pdf = pdftotext.PDF(uploaded_file)
    # s = "\n\n".join(pdf) #join all strings
    # split = s.split('\n')
    # corrected_string = []
    # for line in split:
    #     if len(line.split(' '))>1:
    #         corrected_string.append(line)
    # final_text = "\n".join(corrected_string)
    # #print(final_text)
    # #print(s)   
    # print("Enter any relevant information you think is important for the model to know in order to summarize your text (age, demographics, English reading experience/levels):")
    # demographic = input()
    # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # context = """You will be given an article and some demographic information about a reader. Summarize the article in a way that a user of this demographic finds easy to understand. Demographics: """ + demographic + """ \n Article: """+ final_text + """ Now generate the article summary: """

    # input_ids = tokenizer(context, return_tensors="pt").input_ids
    # gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    # gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # print(gen_text)
    gen_text = "This is a placeholder summary."
    return gen_text

if __name__ == '__main__':
    app.run(debug=True)
