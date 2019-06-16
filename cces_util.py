from lxml import etree

def parse_question_metadata(metadata_filename):
    tree = etree.parse(metadata_filename)
    root = tree.getroot()
    
    # root.nsmap
    all_vars = root.findall(".//{*}var")
    # for var in all_vars:
    #     name = var.get('name')
    #     label = var.find('{*}labl').text
    question_id_to_label = {var.get('name'): var.find('{*}labl').text for var in all_vars}
    return question_id_to_label