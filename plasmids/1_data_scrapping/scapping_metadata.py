import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import time 

# plasmid_ids = np.array([42876, 40786]) 
plasmid_ids = np.load('/Users/anastasiiashcherbakova/Desktop/1_data_scrapping/cleaned_plasmid_ids.npy')

csv_file_path = '/Users/anastasiiashcherbakova/git_projects/masters_project/data/plasmid_metadata.csv'

try:
    data = pd.read_csv(csv_file_path)
    scrapped_ids = np.array(data['Plasmid ID'])
    header_written = True  # Header is already written because the file exists
except FileNotFoundError:
    scrapped_ids = np.array([])
    header_written = False  # This shouldn't occur since you mentioned the file already has data


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
    
    # Setting the metadata to extract
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

csv_file_path = 'plasmid_metadata.csv'
header_written = True

for plasmid_id in plasmid_ids:
    if plasmid_id in scrapped_ids:
        print(f"Skipping plasmid ID {plasmid_id} as it's already processed.")
        continue
    
    metadata = fetch_plasmid_metadata(plasmid_id)
    if metadata:
        df = pd.DataFrame([metadata])
        df.to_csv(csv_file_path, mode='a', header=False, index=True)
        # header_written = True  
        
        print(f"Appended data for plasmid ID {plasmid_id} to {csv_file_path}")
    
    time.sleep(np.random.uniform(3, 5))

print("Metadata collection complete.")
