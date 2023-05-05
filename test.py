import json


features ={'title': '"Attention Is All You Need"', 'A': {}, 'vocabulary': {'Transformer': {'a machine that translates languages'}, 'translate': {'to change words from one language to another'}, 'attention': {'focusing on something'}, 'sentence': {'a group of words that express a complete thought'}, 'English': {'a language spoken in many countries, including the United States and the United Kingdom'}, 'German': {'a language spoken in Germany'}, 'French': {'a language spoken in France'}}, 'BLEU-scores': {'English-to-German': {'28.4'}, 'English-to-French': {'41.0'}}, 'authors': {'Vaswani et al.': {'the people who created the Transformer'}}}
for feature in features:
    print(feature)
    if feature == "title" or feature == "summary":
        continue
    for subfeature in features[feature]:
        print(f"{subfeature}: {features[feature][subfeature]}")