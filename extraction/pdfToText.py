import pdftotext

# Load your PDF
with open("example.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)
print(len(pdf)) #number of pages
for page in pdf:
    print(page)
print("\n\n".join(pdf)) #join all strings