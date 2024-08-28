import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re

plasmid_ids = np.array([42876, 40786]) 
# plasmid_ids = np.load('path_to_your_plasmid_ids.npy')

def clean_text(text):
    """
    Cleans the text by removing excess whitespace, line breaks, and tabs.
    """
    return re.sub(r'\s+', ' ', text).strip()

def fetch_plasmid_metadata(plasmid_id):
    url = f"https://www.addgene.org/{plasmid_id}/"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for plasmid ID {plasmid_id}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    metadata = {
        "Plasmid ID": plasmid_id,
        "Purpose": "Not Available",
        "Depositing Lab": "Not Available",
        "Publication": "Not Available",
        "Backbone": "Not Available",
        "Gene/Insert": "Not Available",
        "Growth in Bacteria": "Not Available",
        "Cloning Information": "Not Available"
    }
    
    # Extracting Purpose and Depositing Lab
    purpose_section = soup.find('ul', {'id': 'plasmid-description-list'})
    if purpose_section:
        purpose_items = purpose_section.find_all('li')
        for item in purpose_items:
            if 'Purpose' in item.text:
                metadata["Purpose"] = clean_text(item.get_text(separator=" ", strip=True)).replace("(How to cite)", "").strip()
            if 'Depositing Lab' in item.text:
                metadata["Depositing Lab"] = clean_text(item.get_text(separator=" ", strip=True)).replace("(How to cite)", "").strip()
    
    # Extracting Publication
    publication_section = soup.find('div', class_='field-label', string='Publication')
    if publication_section:
        pub_elem = publication_section.find_next_sibling('div', class_='field-content')
        if pub_elem:
            pub_link = pub_elem.find('a')
            if pub_link:
                metadata["Publication"] = clean_text(pub_link.get_text(strip=True)).replace("(How to cite)", "").strip()
    
    # Extracting Backbone Information and combining into one field
    backbone_section = soup.find('h2', string='Backbone')
    backbone_details = []
    if backbone_section:
        ul_section = backbone_section.find_next('ul')
        if ul_section:
            li_items = ul_section.find_all('li')
            for li in li_items:
                li_text = clean_text(li.get_text(separator=" ", strip=True))
                backbone_details.append(li_text)
    
    if backbone_details:
        metadata["Backbone"] = "; ".join(backbone_details)
    
    # Extracting Gene/Insert, Growth in Bacteria, Cloning Information and combining into their respective fields
    sections = {
        "Gene/Insert": [],
        "Growth in Bacteria": [],
        "Cloning Information": []
    }
    
    for section_name, details_list in sections.items():
        section_header = soup.find('h2', string=section_name)
        if section_header:
            ul_section = section_header.find_next('ul')
            if ul_section:
                li_items = ul_section.find_all('li')
                for li in li_items:
                    li_text = clean_text(li.get_text(separator=" ", strip=True))
                    details_list.append(li_text)
        
        if details_list:
            metadata[section_name] = "; ".join(details_list)
    
    return metadata

plasmid_metadata_list = []

for plasmid_id in plasmid_ids:
    metadata = fetch_plasmid_metadata(plasmid_id)
    if metadata:
        plasmid_metadata_list.append(metadata)

df = pd.DataFrame(plasmid_metadata_list)
df.to_csv('plasmid_metadata.csv', index=False)

print("Metadata collection complete.")
