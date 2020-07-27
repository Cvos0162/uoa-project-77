from io import StringIO

from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from converter import XMLConverter
from pdfminer.converter import XMLConverter as PDFMinerConverter

outfp = open('../output/sample.xml', "wb")
with open('../sample/Building Regulations 1992 p6.pdf', 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = PDFMinerConverter(rsrcmgr, outfp, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
    device.close()
outfp.close()

outfp = open('../output/output.xml', "wb")
with open('../sample/Building Regulations 1992 p6.pdf', 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = XMLConverter(rsrcmgr, outfp, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
    device.close()
outfp.close()

