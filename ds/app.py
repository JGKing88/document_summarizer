from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
# import pdftotext
import openai
import ds_config
import tokenization
from summarization.summarize import prepare_summary

#Need to create a file config.json in the format
#{
#    "summarization": {"OPENAI_API_KEY": <KEY>,
# "CONTEXT_WINDOW":  <CONTEXT_WINDOW_LENGTH>}
#}
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

    extras = request.form['extras']
    # Process the PDF and user_information to generate a summary
    summary = process_pdf_and_user_information(uploaded_file, user_information, summary_details, extras)

    return jsonify({'summary': summary})

def process_pdf_and_user_information(uploaded_file, user_information="", summary_details="", extras=""):
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

    return prepare_summary(document, user_information, summary_details, extras)


    # return "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)
