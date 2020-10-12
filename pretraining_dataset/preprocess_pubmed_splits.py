__author__ = 'Qiao Jin'

import glob
import json
import xml.etree.ElementTree as ET

for xml_path in glob.glob('pubmed20n*.xml'):
    print('Processing %s' % xml_path)
    output = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for citation in root.iter('MedlineCitation'):
        pmid = citation.find('PMID')
        if pmid == None:
            continue
        else:
            pmid = pmid.text

        texts = []
        sec_labels = []

        title = citation.find('Article/ArticleTitle')
        if title != None:
            texts.append(title.text)
            sec_labels.append('TITLE')
        
        for info in citation.iter('AbstractText'):
            if info.text:
                texts.append(info.text)
                sec_labels.append(info.get('Label'))

        assert len(texts) == len(sec_labels)
        
        output.append({'pmid': pmid,
                       'texts': texts,
                       'sec_labels': sec_labels})

    with open('%s.json' % xml_path.split('.')[0], 'w') as f:
        json.dump(output, f, indent=4)

