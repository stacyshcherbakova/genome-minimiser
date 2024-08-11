import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re

# Example plasmid_ids.npy file loading
plasmid_ids = np.array([42876, 40786])  # Replace with your npy file path
# plasmid_ids = np.load('path_to_your_plasmid_ids.npy')

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
                metadata["Purpose"] = re.sub(r'\s+', ' ', item.get_text(separator=" ", strip=True)).replace("(How to cite)", "").strip()
            if 'Depositing Lab' in item.text:
                metadata["Depositing Lab"] = re.sub(r'\s+', ' ', item.get_text(separator=" ", strip=True)).replace("(How to cite)", "").strip()
    
    # Extracting Publication
    publication_section = soup.find('div', class_='field-label', string='Publication')
    if publication_section:
        pub_elem = publication_section.find_next_sibling('div', class_='field-item')
        if pub_elem:
            metadata["Publication"] = re.sub(r'\s+', ' ', pub_elem.get_text(strip=True)).replace("(How to cite)", "").strip()
    
    # Extracting Backbone
    backbone_section = soup.find('h2', string='Backbone')
    if backbone_section:
        backbone_text = backbone_section.find_next('p').get_text(strip=True)
        metadata["Backbone"] = re.sub(r'\s+', ' ', backbone_text).replace("(How to cite)", "").strip()
    
    # Extracting Gene/Insert
    gene_insert_section = soup.find('h2', string='Gene/Insert')
    if gene_insert_section:
        gene_insert_text = gene_insert_section.find_next('p').get_text(strip=True)
        metadata["Gene/Insert"] = re.sub(r'\s+', ' ', gene_insert_text).replace("(How to cite)", "").strip()
    
    # Extracting Growth in Bacteria
    growth_section = soup.find('h2', string='Growth in Bacteria')
    if growth_section:
        growth_text = growth_section.find_next('p').get_text(strip=True)
        metadata["Growth in Bacteria"] = re.sub(r'\s+', ' ', growth_text).replace("(How to cite)", "").strip()
    
    # Extracting Cloning Information
    cloning_info_section = soup.find('h2', string='Cloning Information')
    if cloning_info_section:
        cloning_info_text = cloning_info_section.find_next('p').get_text(strip=True)
        metadata["Cloning Information"] = re.sub(r'\s+', ' ', cloning_info_text).replace("(How to cite)", "").strip()
    
    return metadata

# Collect metadata for all plasmids
plasmid_metadata_list = []

for plasmid_id in plasmid_ids:
    metadata = fetch_plasmid_metadata(plasmid_id)
    if metadata:
        plasmid_metadata_list.append(metadata)

# Convert to DataFrame and display
df = pd.DataFrame(plasmid_metadata_list)

# Display dataframe
print(df)

# Save the dataframe to a CSV file (optional)
df.to_csv('plasmid_metadata.csv', index=False)

print("Metadata collection complete. Data saved to plasmid_metadata.csv.")