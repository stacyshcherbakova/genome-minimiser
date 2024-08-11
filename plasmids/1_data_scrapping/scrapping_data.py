import os
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse

save_dir = '/home/stacys/data/plasmid_sequences'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

downloaded_plasmids_file = os.path.join(save_dir, 'downloaded_plasmids.txt')

last_processed_page_file = os.path.join(save_dir, 'last_processed_page.txt')

downloaded_plasmids = set()

if os.path.exists(downloaded_plasmids_file):
    with open(downloaded_plasmids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                downloaded_plasmids.add(line)

def is_allowed(url, disallowed_paths):
    parsed_url = urlparse(url)
    for path in disallowed_paths:
        if parsed_url.path.startswith(path):
            return False
    return True

def download_plasmid_sequence(plasmid_id, save_dir):
    '''
    download_plasmid_sequence function downloads plasmid sequence by sending a request to the URL if the full version of the plasmid sequence exists 

    Parameters:
    ----------
    plasmid_id - plasmid id (unique to the AddGene bank)
    save_dir - path to the directory where the plasmid sequences will be stored 
    '''
    
    if plasmid_id in downloaded_plasmids:
        print(f"Plasmid ID {plasmid_id} already processed. Skipping.")
        return

    print(f"Attempting to download sequence for plasmid ID: {plasmid_id}")
    url = f'https://www.addgene.org/{plasmid_id}/sequences/'
    if not is_allowed(url, ['/emta/', '/emta-addgene-public/', '/users/login/']):
        print(f"Access to {url} is disallowed by robots.txt")
        return

    for attempt in range(5):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            h2_tags = soup.find_all('h2')
            for h2 in h2_tags:
                if 'Full Sequences from' in h2.text:
                    textarea = h2.find_next('textarea', {'class': 'copy-from form-control'})
                    if textarea:
                        sequence_text = textarea.text.strip()
                        file_path = os.path.join(save_dir, f'{plasmid_id}.txt')
                        with open(file_path, 'w') as f:
                            f.write(sequence_text)
                        downloaded_plasmids.add(plasmid_id)
                        with open(downloaded_plasmids_file, 'a') as f:
                            f.write(f'{plasmid_id}\n')
                        print(f'Successfully downloaded sequence for: {plasmid_id}')
                        return
            print(f'Sequence not found for: {plasmid_id}')
            return
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 4:
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to download sequence for: {plasmid_id} after 5 attempts.")
                return

def process_search_results(search_url):
    '''
    process_search_results function processes each page with Bacterial 
    Expression plasmids by scraping the website and finding plasmid IDs

    Parameters:
    ----------
    search_url - current page search results 
    '''

    print(f"Processing search results from URL: {search_url}")
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            if href.startswith('/') and href[1:-1].isdigit() and href.endswith('/'):
                plasmid_id = href.strip('/').split('/')[0]
                print(f'Found plasmid ID: {plasmid_id}')
                download_plasmid_sequence(plasmid_id, save_dir)
    else:
        print(f'Failed to access search results URL: {search_url}')

def save_last_processed_page(page_number):
    with open(last_processed_page_file, 'w') as f:
        f.write(str(page_number))

def load_last_processed_page():
    if os.path.exists(last_processed_page_file):
        with open(last_processed_page_file, 'r') as f:
            return int(f.read().strip())
    return 1  

def main():
    '''
    main function iterates over every page with Bacterial Expression plasmids search results
    '''

    # Page size has to be 20 for the script to run on all 600 pages and not miss any plasmids 
    base_url = 'https://www.addgene.org/search/catalog/plasmids/?q=&page_size=20&expression=Bacterial+Expression'
    
    # Use if scripts tends to halt to start from the last seen page
    # last_processed_page = load_last_processed_page()
    # print(f"Starting from page {last_processed_page}.")

    # Has to be 600, if chose page size 50 change to ~250 (check Addgene website)
    total_pages = 600
    for page in range(1, total_pages + 1):
        paginated_url = f'{base_url}&page_number={page}'
        print(f"Processing page {page}.")
        process_search_results(paginated_url)
        save_last_processed_page(page)
        time.sleep(5)  

# Parallel processing
if __name__ == '__main__':
    main()
