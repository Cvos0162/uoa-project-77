from io import StringIO

from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from LegalDocMLconverter import LegalDocMLconverter
from pdfminer.converter import XMLConverter as PDFMinerConverter

input_path = "../sample/e2-external-moisture-3rd-edition-amendment-8.pdf"
output_path = "../output/output.xml"
output_path2 = "../output/sample.xml"

img_output_path = "../output/pdfminer_page1.jpg"

outfp = open(output_path, "wb")
with open(input_path, 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = LegalDocMLconverter(rsrcmgr, outfp, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
    #device.draw_layout(input_path, img_output_path)
    device.close()
outfp.close()

#outfp = open(output_path2, "wb")
#with open(input_path, 'rb') as in_file:
#    parser = PDFParser(in_file)
#    doc = PDFDocument(parser)
#    rsrcmgr = PDFResourceManager()
#    device = PDFMinerConverter(rsrcmgr, outfp, laparams=LAParams())
#    interpreter = PDFPageInterpreter(rsrcmgr, device)
#    for page in PDFPage.create_pages(doc):
#        interpreter.process_page(page)
#    device.close()
#outfp.close()

