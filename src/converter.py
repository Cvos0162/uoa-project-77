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
from pdfminer.utils import apply_matrix_pt
from pdfminer.utils import mult_matrix
from pdfminer.utils import enc
from pdfminer.utils import bbox2str
from pdfminer import utils

from pdfminer.converter import PDFConverter

class XMLConverter(PDFConverter):

    CONTROL = re.compile('[\x00-\x08\x0b-\x0c\x0e-\x1f]')

    def __init__(self, rsrcmgr, outfp, codec='utf-8', pageno=1, laparams=None,
                 imagewriter=None, stripcontrol=False):
        PDFConverter.__init__(self, rsrcmgr, outfp, codec=codec, pageno=pageno,
                              laparams=laparams)
        self.imagewriter = imagewriter
        self.stripcontrol = stripcontrol
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

        def extract_textbox(item):
            if isinstance(item, LTPage):
                for child in item:
                    extract_textbox(child)
            elif isinstance(item, LTRect):
                for child in item:
                    extract_textbox(child)
            elif isinstance(item, LTFigure):
                for child in item:
                    extract_textbox(child)
            elif isinstance(item, LTTextLine):
                for child in item:
                    extract_textbox(child)
            elif isinstance(item, LTTextBox):
                self.items.append(item)
                for child in item:
                    extract_textbox(child)

        extract_textbox(ltpage)

        def get_y0(item):
            return item.y0

        self.items.sort(key=get_y0, reverse=True)

        def group_textboxes(items):
            new_items = []
            prev = items[0]
            for item in items[1:]:
                y_diff = (prev.y0 - item.y1)
                x_diff = (item.x0 - prev.x0)
                if y_diff < 3 and x_diff < 31 and x_diff > 0:
                    xs = [item.x0, item.x1, prev.x0, prev.x1]
                    ys = [item.y0, item.y1, prev.y0, prev.y1]
                    prev.add(item)
                    prev.set_bbox((min(xs), min(ys), max(xs), max(ys)))
                else:
                    new_items.append(prev)
                    prev = item
            return new_items

        def render(item):
            if isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'
                s = '<textbox id="%d" bbox="%s"%s>\n' %\
                    (item.index, bbox2str(item.bbox), wmode)
                self.write(s)
                #self.write(item.get_text())
                for child in item:
                    self.write(highlights(child))
                self.write('</textbox>\n')
            else:
                assert False, str(('Unhandled', item))
            
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

            """
            if isinstance(item, LTPage):
                s = '<page id="%s" bbox="%s" rotate="%d">\n' % \
                    (item.pageid, bbox2str(item.bbox), item.rotate)
                self.write(s)
                for child in item:
                    render(child)
                if item.groups is not None:
                    self.write('<layout>\n')
                    for group in item.groups:
                        show_group(group)
                    self.write('</layout>\n')
                self.write('</page>\n')
            elif isinstance(item, LTLine):
                s = '<line linewidth="%d" bbox="%s" />\n' % \
                    (item.linewidth, bbox2str(item.bbox))
                self.write(s)
            elif isinstance(item, LTRect):
                s = '<rect linewidth="%d" bbox="%s" />\n' % \
                    (item.linewidth, bbox2str(item.bbox))
                self.write(s)
            elif isinstance(item, LTCurve):
                s = '<curve linewidth="%d" bbox="%s" pts="%s"/>\n' % \
                    (item.linewidth, bbox2str(item.bbox), item.get_pts())
                self.write(s)
            elif isinstance(item, LTFigure):
                s = '<figure name="%s" bbox="%s">\n' % \
                    (item.name, bbox2str(item.bbox))
                self.write(s)
                for child in item:
                    render(child)
                self.write('</figure>\n')
            elif isinstance(item, LTTextLine):
                self.write('<textline bbox="%s">\n' % bbox2str(item.bbox))
                for child in item:
                    render(child)
                self.write('</textline>\n')
            elif isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'
                s = '<textbox id="%d" bbox="%s"%s>\n' %\
                    (item.index, bbox2str(item.bbox), wmode)
                self.write(s)
                for child in item:
                    render(child)
                self.write('</textbox>\n')
            elif isinstance(item, LTChar):
                s = '<text font="%s" bbox="%s" colourspace="%s" ' \
                    'ncolour="%s" size="%.3f">' % \
                    (enc(item.fontname), bbox2str(item.bbox),
                     item.ncs.name, item.graphicstate.ncolor, item.size)
                self.write(s)
                self.write_text(item.get_text())
                self.write('</text>\n')
            elif isinstance(item, LTText):
                self.write('<text>%s</text>\n' % item.get_text())
            elif isinstance(item, LTImage):
                if self.imagewriter is not None:
                    name = self.imagewriter.export_image(item)
                    self.write('<image src="%s" width="%d" height="%d" />\n' %
                               (enc(name), item.width, item.height))
                else:
                    self.write('<image width="%d" height="%d" />\n' %
                               (item.width, item.height))
            else:
                assert False, str(('Unhandled', item))
            return
            """
        textboxes = group_textboxes(self.items)
        for item in textboxes:
            render(item)
        return

    def close(self):
        self.write_footer()
        return