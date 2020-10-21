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
from pdfminer.layout import LTTextLine
from pdfminer.layout import LTTextBox
from pdfminer.layout import LTTextBoxVertical
from pdfminer.layout import LTTextGroup
from pdfminer.layout import LTAnno
from pdfminer.utils import apply_matrix_pt
from pdfminer.utils import mult_matrix
from pdfminer.utils import enc
from pdfminer.utils import bbox2str
from pdfminer import utils

from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
import pickle
from treelib import Node, Tree

from pdfminer.converter import PDFConverter
from NLPTextBox import NLPTextBox, NLPSimpleBox

class LegalDocMLconverter(PDFConverter):

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
        self.classified = []
        self.classified_header = []
        self.classified_paragraph = []
        self.classified_section = []
        self.classified_subsection = []
        self.tree = Tree()
        self.tree.create_node("Documents", 'documents')
        self.num_tabs = 0
        self.write_header()

        self.headerExist = False
        self.in_li = False

        json_file = open('data/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("data/model.h5")
        
        self.tokenizer = []

        with open('data/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        return

    def decode_tags(self, pred) :
        tags = {
            'header':0,
            'document':1,
            'paragraph':2,
            'topic':3,
            'section':4,
            'subsection':5,
            'li':6,
            'footer':7,
            'page_number':8,
            'figure':9,
            'table':10,
            'table_li':11,
            'commentary':12,
            '?':13,
        }
        decode = {v: k for k, v in tags.items()}
        num_tags = max(tags.values()) + 1
        
        return decode[np.argmax(pred)]

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
        self.write('<documents>\n')
        self.num_tabs = 1
        return

    def write_footer(self):
        self.write('</documents>\n')
        self.num_tabs = 0
        return

    def write_text(self, text):
        if self.stripcontrol:
            text = self.CONTROL.sub('', text)
        self.write(enc(text))
        return

    def write_tab(self):
        for i in range(self.num_tabs):
            self.write("\t")

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

        def get_id(item):
            return item.index

        def get_size(item):
            if isinstance(item, LTChar):
                return item.size
            elif isinstance(item, LTAnno):
                return 0
            else:
                for child in item:
                    return get_size(child)

        self.items.sort(key=get_y0, reverse=True)

        def group_textboxes(items):
            new_items = []
            prev = items[0]
            for item in items[1:]:
                if isinstance(prev, LTChar):
                    box = LTTextBox()
                    box.add(prev)
                    box.set_bbox((prev.x0, prev.y0, prev.x1, prev.y1))
                    prev = box
                y_diff = (prev.y0 - item.y1)
                x_diff = (item.x0 - prev.x1)
                if y_diff < get_size(prev)/2 and x_diff < get_size(prev) and x_diff >= -get_size(prev)/2:
                    xs = [item.x0, item.x1, prev.x0, prev.x1]
                    ys = [item.y0, item.y1, prev.y0, prev.y1]
                    prev.add(item)
                    prev.set_bbox((min(xs), min(ys), max(xs), max(ys)))
                elif y_diff < get_size(prev)/2 and (item.x0 - prev.x0) < get_size(prev)/2 and (item.x1 - prev.x1) > -get_size(prev)/2:
                    vert = LTTextBoxVertical()
                    xs = [item.x0, item.x1, prev.x0, prev.x1]
                    ys = [item.y0, item.y1, prev.y0, prev.y1]
                    for child in prev:
                        vert.add(child)
                    vert.add(item)
                    vert.set_bbox((min(xs), min(ys), max(xs), max(ys)))
                    prev = vert
                else:
                    new_items.append(prev)
                    prev = item
                #new_items.append(prev)
                #prev = item
            new_items.append(prev)
            return new_items

        def classify(item):
            if isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'

                box = NLPTextBox(item)
                
                s = ('%s %d %d %d ' % (bbox2str(box.bbox), box.b, box.i, box.size) + item.get_text().replace('\n', ' '))
                
                s_list = []
                s_list.append(s)
                X = self.tokenizer.texts_to_sequences(s_list)
                maxlen = 100
                X = pad_sequences(X, padding='post', maxlen=maxlen)
                preds = self.model.predict(X)
                tag = self.decode_tags(preds)

                box.set_tag(tag)

                if (tag == "header"):
                    self.classified_header.append(box)
                elif (tag == "paragraph"):
                    self.classified_paragraph.append(box)
                elif (tag == "section"):
                    self.classified_section.append(box)
                elif (tag == "subsection"):
                    self.classified_subsection.append(box)

                self.classified.append(box)
                    
            else:
                assert False, str(('Unhandled', item))

        def into_tree():
            _header = self.classified_header[0]
            self.tree.create_node(_header.get_text(), _header.key, parent="documents", data=_header)

            for _section in self.classified_section:
                self.tree.create_node(_section.get_text(), _section.key, parent=_header.key, data=_section)

            for _sebsection in self.classified_subsection:
                keys = _sebsection.key.split('.')
                keys.pop()
                _key = ''.join([i + "." for i in keys])
                _key = _key[:-1] + ".0"
                if (not self.tree.contains(_key)):
                    data = NLPSimpleBox("section", _key)
                    self.tree.create_node(_key, _key, parent=_header.key, data=data)
                    self.classified.append(data)
                self.tree.create_node(_sebsection.get_text(), _sebsection.key, parent=_key, data=_sebsection)

            for _paragraph in self.classified_paragraph:
                keys = _paragraph.key.split('.')
                keys.pop()
                _key = ''.join([i + "." for i in keys])
                _key = _key[:-1]
                if (not self.tree.contains(_key)):
                    section_keys = _key.split('.')
                    section_keys.pop()
                    section_key = ''.join([i + "." for i in section_keys])
                    section_key = section_key[:-1] + ".0"

                    if (not self.tree.contains(section_key)):
                        data = NLPSimpleBox("section", _key)
                        self.tree.create_node(section_key, section_key, parent=_header.key, data=data)
                        self.classified.append(data)

                    data = NLPSimpleBox("subsection", _key)
                    self.tree.create_node(_key, _key, parent=section_key, data=data)
                    self.classified.append(data)

                self.tree.create_node(_paragraph.get_text(), _paragraph.key, parent=_key, data=_paragraph)

            new_classified = []

            prev_box = self.classified[0]
            for _boxes in self.classified:
                if _boxes.tag == "commentary":
                    if prev_box.tag == "commentary":
                        prev_box.text += _boxes.text
                        prev_box.set_tag("commentary")
                    else:
                        prev_box = _boxes
                else:
                    if prev_box.tag == "commentary":
                        new_classified.append(prev_box)
                    prev_box = _boxes
                    new_classified.append(_boxes)

            self.classified = new_classified

            for _boxes in self.classified:
                
                if (_boxes.tag == "footer"):
                    None
                elif (_boxes.tag == "page_number"):
                    None
                elif (_boxes.tag == "?"):
                    None
                elif (_boxes.tag == "topic"):
                    _prev_subsection = find_prev_with_tag(_boxes, "subsection")
                    try:
                        self.tree.create_node(_boxes.get_text(), _boxes.key, parent=_prev_subsection.key, data=_boxes)
                    except:
                        self.tree.create_node(_boxes.get_text(), _boxes.key  + '1', parent=_prev_subsection.key, data=_boxes)
        
                elif _boxes.tag != "header" and _boxes.tag != "paragraph" and _boxes.tag != "section" and _boxes.tag != "subsection":
                    _prev_paragraph = find_prev_with_tag(_boxes, "paragraph")
                    try:
                        self.tree.create_node(_boxes.get_text(), _prev_paragraph.key + "." + _boxes.key, parent=_prev_paragraph.key, data=_boxes)
                    except: 
                        self.tree.create_node(_boxes.get_text(), _prev_paragraph.key + "." + _boxes.key + '1', parent=_prev_paragraph.key, data=_boxes)
                    
        def find_prev_with_tag(item, tag):
            _prev = ''
            _next = False
            for _boxes in self.classified:
                if (_boxes.tag == tag):
                    _prev = _boxes
                    if (_next):
                        break
                if (_boxes == item):
                    if (_prev == ''):
                        _next = True
                    else:
                        break
            return _prev

        def get_node_id(node):
            return node.identifier

        def render(node):
            
            tag = ''
            item = node.data
            
            if isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'
                
                tag = item.tag

                if (tag == "header"):
                    if (not self.headerExist):
                        self.write_tab()
                        self.write('<document title="%s">\n' % item.get_text())     
                        self.num_tabs = self.num_tabs + 1
                        self.headerExist = True

                elif (tag == "paragraph"):
                    self.write_tab()
                    self.write('<paragraph key="%s">\n' % item.get_key())
                    self.num_tabs = self.num_tabs + 1
                    self.write_tab()
                    self.write("<p>" + item.get_text().replace('\n', ' ').lstrip().rstrip() + "</p>\n")

                elif (tag == "commentary"):
                    self.write_tab()
                    self.write('<commentary title="COMMENT:">')
                    self.write(item.get_text().replace('COMMENT:', '').lstrip())
                    self.write('</commentary>\n')

                elif (tag == "topic"):
                    self.write_tab()
                    self.write('<topic>')
                    self.write(item.get_text())
                    self.write('</topic>\n')

                elif (tag == "section"):
                    self.write_tab()
                    self.write('<section key="%s" title="%s">\n' % (item.get_key(), item.get_text()))
                    self.num_tabs = self.num_tabs + 1

                elif (tag == "subsection"):
                    self.write_tab()
                    self.write('<subsection key="%s" title="%s">\n' % (item.get_key(), item.get_text()))
                    self.num_tabs = self.num_tabs + 1

                elif (tag == "li"):
                    if (not self.in_li):
                        self.write_tab()
                        self.write('<ol>\n')
                        self.num_tabs = self.num_tabs + 1
                        self.in_li = True
                    self.write_tab()
                    if (item.list_tag):
                        self.write('<li key="%s">' % node.identifier)
                        self.write(item.get_text())
                        self.write('</li>\n')
                    else:
                        self.write('<li>')
                        self.write(item.get_text())
                        self.write('</li>\n')

                elif (tag == "footer"):
                    None
                elif (tag == "page_number"):
                    None
                elif (tag == "?"):
                    None
                else:
                    self.write(self.decode_tags(preds)+ s + "\n")
            

            branches = self.tree.is_branch(node.identifier)
            _branches = []
            for child in branches:
                _branches.append(self.tree.get_node(child))

            if (tag == "section" or tag == "subsection"):
                _branches.sort(key=get_node_id, reverse=False)

            for _child in _branches:
                render(_child)
            
            if (tag != "li" and tag != "footer" and tag != "page_number" and tag != "?" and self.in_li):
                self.num_tabs = self.num_tabs - 1
                self.write_tab()
                self.write('</ol>\n')
                self.in_li = False

            if (tag == "paragraph"):
                self.num_tabs = self.num_tabs - 1
                self.write_tab()
                self.write('</paragraph>\n')
            elif (tag == "header"):
                self.num_tabs = self.num_tabs - 1
                self.write_tab()
                self.write('</document>\n')
            elif (tag == "section"):
                self.num_tabs = self.num_tabs - 1
                self.write_tab()
                self.write('</section>\n')
            elif (tag == "subsection"):
                self.num_tabs = self.num_tabs - 1
                self.write_tab()
                self.write('</subsection>\n')  

        def highlights(item):
            s = ''
            prev_bold = False
            prev_italic = False
            for child in item:
                if isinstance(child, LTChar):
                    if 'Bold' in child.fontname:
                        if prev_italic:
                            s += '</i>'
                        if not prev_bold:
                            s += '<b>'
                            
                        prev_bold = True
                        prev_italic = False
                    elif 'Italic' in child.fontname:
                        if prev_bold:
                            s += '</b>'
                        if not prev_italic:
                            s += '<i>'
                        
                        prev_italic = True
                        prev_bold = False
                    else:
                        if prev_bold:
                            s += '</b>'
                        elif prev_italic:
                            s += '</i>'
                        prev_bold = False
                        prev_italic = False
                    
                    s += child.get_text()
                
                elif isinstance(child, LTTextLine):
                    s += highlights(child)
                elif isinstance(child, LTTextBox):
                    s += highlights(child)
                elif isinstance(child, NLPTextBox):
                    s += highlights(child)
                else:
                    if child.get_text() == '\n':
                        if prev_bold:
                            s += '</b>'
                        elif prev_italic:
                            s += '</i>'
                        
                        prev_bold = False
                        prev_italic = False
                    s +=  child.get_text()
            return s

        self.textboxes = group_textboxes(self.items)
        
        self.textboxes.sort(key=get_id, reverse=False)
        for item in self.textboxes:
            classify(item)
        
        into_tree()

        self.tree.show()

        render(self.tree.get_node("documents"))

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
        for item in self.textboxes:
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