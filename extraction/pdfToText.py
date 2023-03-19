import pdftotext

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
print(final_text)
#print(s)


