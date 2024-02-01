import magic

m = magic.from_buffer("computerForensics.docx")
m2 = magic.from_file("computerForensics.docx")
print('File Signature:: ', m)
print('File Signature Text:: ', m2)
