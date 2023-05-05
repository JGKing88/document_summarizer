from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pdftotext
import openai
import ds_config
import tokenization
from summarization.summarize import prepare_summary
import json


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
    # final_dict = {"summary": " At a party, Gabriel finds himself partnered with Miss Ivors, a talkative young lady. They discuss their careers, and Miss Ivors invites Gabriel and his wife Gretta to join her on a trip to the Aran Isles. Gabriel declines, saying he has plans to go on a cycling tour in Europe. Miss Ivors criticizes him for not visiting his own country and not knowing the Irish language. Later, Gabriel carves a roast goose for dinner, and the guests discuss the opera company in town. Gabriel feels uneasy about his earlier conversation with Miss Ivors.  Gabriel and Gretta attend a party, and later, in their hotel room, Gretta shares a memory of a young man named Michael Furey who used to sing 'The Lass of Aughrim' and died when he was only seventeen. Gabriel realizes that he never truly knew his wife and reflects on love and the passage of time as snow falls outside."}
    summary, features = prepare_summary(document, user_information, summary_details, extras)
    final_dict = {"summary": summary, "features": features}

    return json.dumps(final_dict)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)
