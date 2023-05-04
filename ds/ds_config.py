import json

def api_key():
    file = open("config.json")
    data = json.load(file)
    return data["summarization"]["API_KEY"]

