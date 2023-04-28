from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
# import pdftotext
import openai
import ds_config
import tokenization

#Need to create a file ds_config.py that returns the api_key as a string on calling api_key()
openai.api_key = ds_config.api_key() 

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    uploaded_file = request.files['pdf_file']
    user_information = request.form['user_information']
    summary_details = request.form['summary_details']
    
    # Process the PDF and user_information to generate a summary
    summary = process_pdf_and_user_information(uploaded_file, user_information, summary_details)

    return jsonify({'summary': summary})

def process_pdf_and_user_information(uploaded_file, user_information="", summary_details=""):
    # Placeholder function for processing the PDF and user_information
    # You can implement your own summary generation logic here
    pdf = pdftotext.PDF(uploaded_file)
    s = "\n\n".join(pdf) #join all strings
    split = s.split('\n')
    corrected_string = []
    for line in split:
        if len(line.split(' '))>1:
            corrected_string.append(line)
    document = "\n".join(corrected_string)
    tokenized_document = tokenization.tok_doc(document)
    max_sum_len = tokenization.max_sum_len(tokenized_document)
    #print(final_text)
    #print(s)

    #document = ""
    #max_sum_len = 64

    prompt_prompt = "Generate a prompt that tells ChatGPT to generate a summary of the following document for this type of user: " + user_information + " and this type of summary: " + summary_details
    prompt = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "user", "content": prompt_prompt},
            ],
        temperature=0.2, #[0,2] higher values are more random (want low randomnes)
        max_tokens=max(1000, max_sum_len),
    )

    print(prompt.choices[0].message.content)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "user", "content": prompt.choices[0].message.content + "\n" + document},
            ],
        temperature=0.7, #higher values are more random
        max_tokens=max(1000, max_sum_len),
        frequency_penalty=0.0, #[-2, 2] Positive values decrease the model's likelihood to repeat the same line verbatim.
        presence_penalty=-1.0 #[-2, 2] Positive values increase the model's likelihood to talk about new topics.
    )

    gen_text = response.choices[0].message.content
    return gen_text

    # return "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)
