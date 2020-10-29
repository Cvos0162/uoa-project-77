import logging
import re
import sys
from pdfminer.pdfdevice import PDFTextDevice
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.layout import LTContainer
from pdfminer.layout import LTPage
from pdfminer.layout import LTText
from pdfminer.layout import LTLine
from pdfminer.layout import LTRect
from pdfminer.layout import LTCurve
from pdfminer.layout import LTFigure
from pdfminer.layout import LTImage
from pdfminer.layout import LTChar
from pdfminer.layout import LTAnno
from pdfminer.layout import LTTextLine
from pdfminer.layout import LTTextBox
from pdfminer.layout import LTTextBoxVertical
from pdfminer.layout import LTTextGroup
from pdfminer.utils import apply_matrix_pt
from pdfminer.utils import mult_matrix
from pdfminer.utils import enc
from pdfminer.utils import bbox2str
from pdfminer import utils

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

from pdfminer.converter import PDFConverter

class PDFMinerConverter(PDFConverter):

    CONTROL = re.compile('[\x00-\x08\x0b-\x0c\x0e-\x1f]')

    def __init__(self, rsrcmgr, outfp, codec='utf-8', pageno=1, laparams=None,
                 imagewriter=None, stripcontrol=False):
        PDFConverter.__init__(self, rsrcmgr, outfp, codec=codec, pageno=pageno,
                              laparams=laparams)
        self.imagewriter = imagewriter
        self.stripcontrol = stripcontrol
        self.textboxes = []
        self.page_width = []
        self.page_height = []
        self.write_header()
        return

    def write(self, text):
        if self.codec:
            text = text.encode(self.codec)
        self.outfp.write(text)
        return

    def write_header(self):
        if self.codec:
            self.write('<?xml version="1.0" encoding="%s" ?>\n' % self.codec)
        else:
            self.write('<?xml version="1.0" ?>\n')
        self.write('<pages>\n')
        return

    def write_footer(self):
        self.write('</pages>\n')
        return

    def write_text(self, text):
        if self.stripcontrol:
            text = self.CONTROL.sub('', text)
        self.write(enc(text))
        return

    def receive_layout(self, ltpage):
        self.items = []

        def extract_text(item):
            if isinstance(item, LTPage):
                #print(bbox2str(item.bbox))
                self.page_width = item.x1
                self.page_height = item.y1
                for child in item:
                    extract_text(child)
            elif isinstance(item, LTFigure):
                for child in item:
                    extract_text(child)
            elif isinstance(item, LTTextBox):
                self.items.append(item)
            elif isinstance(item, LTChar):
                self.items.append(item)

        extract_text(ltpage)

        def get_y0(item):
            return item.y0

        def get_size(item):
            if isinstance(item, LTChar):
                return item.size
            elif isinstance(item, LTAnno):
                return 0
            else:
                for child in item:
                    return get_size(child)

        self.items.sort(key=get_y0, reverse=True)

        def render(item):
            if isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'
                s = '<textbox id="%d" bbox="%s"%s>\n' %\
                    (item.index, bbox2str(item.bbox), wmode)
                self.write(s)
                self.write(item.get_text())
                self.write('</textbox>\n')
            else:
                assert False, str(('Unhandled', item))

        for item in self.items:
            render(item)
        return

    def draw_layout(self, input_path, output_path):
        #init cv2

        pages = convert_from_path(input_path, 500)
        
        pages[0].save(output_path, 'JPEG')
        page1 = cv2.imread(output_path)

        page1_disp = page1
        for i in range(3):
            page1_disp = cv2.pyrDown(page1_disp)

        height, width, channels = page1.shape
        #print(width, height)
        #print(height)
        scale = height/int(self.page_height)
        for item in self.items:
            if isinstance(item, LTTextBox) or isinstance(item, LTChar):
                #render cv2
                
                start = (int(item.x0 * scale), (height - int(item.y0 * scale)))
                end = (int(item.x1 * scale), (height - int(item.y1 * scale)))
                #print(start , end)
                color = (0, 0, 255)
                thickness = 5
                page1 = cv2.rectangle(page1, start, end, color, thickness)
            else:
                assert False, str(('Unhandled', item))

        page1 = cv2.rectangle(page1, (40,40), (50,50), (0,0,255), 2)
        boxed_disp = page1
        for i in range(3):
            boxed_disp = cv2.pyrDown(boxed_disp)

        while True:
            cv2.imshow('page', page1_disp)
            cv2.imshow('boxed', boxed_disp)
        
            #exit on ESC
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            
        cv2.destroyAllWindows()

    def close(self):
        self.write_footer()
        return