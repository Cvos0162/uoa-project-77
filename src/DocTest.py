from difflib import SequenceMatcher
from bs4 import BeautifulSoup as Soup
import re


def text_cleanup(text):
    return ' '.join(text.replace('\n', ' ').lstrip().rstrip().split())


def text_similarity(text):
    return

new_soup = Soup(open("../output/output.xml"), features="xml")
truth_soup = Soup(open("../sample/NZBC-G4v3.4text.xml"), features="xml")

print("test accuracy of tagging")

number_section = 0
accurate_section_tagging = 0
for tag in new_soup.findAll('section'):
    number_section = number_section + 1
    found = False
    for truth in truth_soup.findAll(key=tag['key']):
        if (text_cleanup(truth['title']) == text_cleanup(tag['title'])):
            found = True
    if found:
        accurate_section_tagging = accurate_section_tagging + 1
    else:
        print("-----section-----")
        print(tag['key'], tag['title'])
        for truth in truth_soup.findAll(key=tag['key']):
            print(truth['title'])

number_subsection = 0
accurate_subsection_tagging = 0
for tag in new_soup.findAll('subsection'):
    number_subsection = number_subsection + 1
    found = False
    for truth in truth_soup.findAll(key=tag['key']):
        if (text_cleanup(truth['title']) == text_cleanup(tag['title'])):
            found = True
    if found:
        accurate_subsection_tagging = accurate_subsection_tagging + 1
    else:
        print("-----subsection-----")
        print(tag['title'])
        for truth in truth_soup.findAll(key=tag['key']):
            print(truth['title'])

number_paragraph = 0
accurate_paragraph_tagging = 0
for tag in new_soup.findAll('paragraph'):
    number_paragraph = number_paragraph + 1
    found = False
    for truth in truth_soup.findAll(key=tag['key']):
        if (text_cleanup(truth.p.string) == text_cleanup(tag.p.string)):
            found = True
    if found:
        accurate_paragraph_tagging = accurate_paragraph_tagging + 1
    else:
        print("-----paragraph-----")
        print(tag['key'], tag.p.string)
        for truth in truth_soup.findAll(key=tag['key']):
            print(truth.p.string)

number_list = 0
accurate_list_tagging = 0
for tag in new_soup.findAll('li'):
    number_list = number_list + 1
    found = False
    if (tag.get('key') != None):
        for truth in truth_soup.findAll(key=tag['key']):
            try:
                if (re.search(text_cleanup(tag.string), text_cleanup(truth.string)) is not None) or (text_cleanup(tag.string) == text_cleanup(truth.string)):
                    found = True
            except:
                try:
                    if (re.search(text_cleanup(tag.get_text()), text_cleanup(truth.tag.get_text())) is not None) or (text_cleanup(tag.tag.get_text()) == text_cleanup(truth.tag.get_text())):
                        found = True
                except:
                    try:
                        for truth in truth_soup.findAll(text=re.compile('.(%s)' % tag.string)):
                            found = True
                    except:
                        for truth in truth_soup.findAll(text=re.compile('.(%s)' % re.escape(tag.string))):
                            found = True
                    if found:
                        accurate_list_tagging = accurate_list_tagging + 1
                    else:
                        print("-----li-----")
                        print(tag.string)
                        None
        if found:
            accurate_list_tagging = accurate_list_tagging + 1
        else:
            print("-----li-----")
            print(text_cleanup(tag.string))
            for truth in truth_soup.findAll(key=tag['key']):
                try:
                    print(text_cleanup(truth.string))
                except:
                    print(text_cleanup(truth.get_text()))
    else:
        try:
            for truth in truth_soup.findAll(text=re.compile('.(%s)' % tag.string)):
                found = True
        except:
            for truth in truth_soup.findAll(text=re.compile('.(%s)' % re.escape(tag.string))):
                found = True
        if found:
            accurate_list_tagging = accurate_list_tagging + 1
        else:
            print("-----li-----")
            print(tag.string)
            None

number_topic = 0
accurate_topic_tagging = 0
for tag in new_soup.findAll('topic'):
    number_topic = number_topic + 1
    found = False
    for truth in truth_soup.findAll('topic'):
        try:
            if (re.search(text_cleanup(tag.string), text_cleanup(truth.string)) is not None) or (text_cleanup(tag.string) == text_cleanup(truth.string)):
                found = True
        except:
            if (re.search(text_cleanup(tag.getText()), text_cleanup(truth.getText())) is not None) or (text_cleanup(tag.getText()) == text_cleanup(truth.getText())):
                found = True
    if found:
        accurate_topic_tagging = accurate_topic_tagging + 1
    else:
        print("-----topic-----")
        print(tag.get_text())
        None

number_comment = 0
accurate_comment_tagging = 0
for tag in new_soup.findAll('commentary'):
    number_comment = number_comment + 1
    found = False
    if (tag.get_text() is not None):
        for truth in truth_soup.findAll('commentary'):
            try:
                if (re.search(text_cleanup(tag.get_text()), text_cleanup(truth.get_text())) is not None) or (text_cleanup(tag.get_text()) == text_cleanup(truth.get_text())):
                    found = True
            except:
                if (re.search(re.escape(text_cleanup(tag.get_text())), re.escape(text_cleanup(truth.get_text()))) is not None) or (text_cleanup(tag.get_text()) == text_cleanup(truth.get_text())):
                    found = True
        if found:
            accurate_comment_tagging = accurate_comment_tagging + 1
        else:
            print("-----commentary-----")
            print(tag.get_text())
            None
    else:
        number_comment = number_comment - 1

print(
    accurate_section_tagging,
    accurate_subsection_tagging,
    accurate_paragraph_tagging,
    accurate_list_tagging,
    accurate_topic_tagging,
    accurate_comment_tagging
)
print(
    number_section,
    number_subsection,
    number_paragraph,
    number_list,
    number_topic,
    number_comment
)

all_tagging = accurate_section_tagging + accurate_subsection_tagging + accurate_paragraph_tagging + \
    accurate_list_tagging + accurate_topic_tagging + accurate_comment_tagging
all_number = number_section + number_subsection + \
    number_paragraph + number_list + number_topic + number_comment

print(str(all_tagging) + "/" + str(all_number) +
      " = " + str(all_tagging/all_number))


print("test accuracy of text")

similarity_in = []
similarity_out = []
for tag in new_soup.findAll('paragraph'):
    for truth in truth_soup.findAll(key=tag['key']):
        similarity_in.append(text_cleanup(tag.get_text()))
        similarity_out.append(text_cleanup(truth.get_text()))

similarity_in = ' '.join(similarity_in)
similarity_out = ' '.join(similarity_out)

s = SequenceMatcher(None, similarity_in, similarity_out)

missing = 0
for tag, i1, i2, j1, j2 in s.get_opcodes():
    if tag == 'replace' or tag == 'delete':
        missing = missing + (i2 - i1)
        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag,
                                                                     i1, i2, j1, j2, similarity_in[i1:i2], similarity_out[j1:j2]))

print(str(len(similarity_in) - missing) + "/" + str(len(similarity_in)))
print((len(similarity_in) - missing)/len(similarity_in))
