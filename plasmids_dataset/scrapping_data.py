import os
import requests
from bs4 import BeautifulSoup

# Directory to save plasmid sequences
save_dir = '/Users/anastasiiashcherbakova/git_projects/masters_project/data/plasmid_sequences'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to download plasmid sequence
def download_plasmid_sequence(plasmid_id, save_dir):
    print(f"Attempting to download sequence for plasmid ID: {plasmid_id}")
    url = f'https://www.addgene.org/{plasmid_id}/sequences/'
    response = requests.get(url)
    if response.status_code == 200:
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
                    print(f'Successfully downloaded sequence for: {plasmid_id}')
                    return
        print(f'Sequence not found for: {plasmid_id}')
    else:
        print(f'Failed to access page for: {plasmid_id}')

# Function to process the search results page and download plasmid sequences
def process_search_results(search_url):
    print(f"Processing search results from URL: {search_url}")
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            if href.startswith('/') and href[1:-1].isdigit() and href.endswith('/'):
                plasmid_id = href.strip('/').split('/')[0]
                plasmid_name = link.text.strip()
                print(f'Found plasmid ID: {plasmid_id}, Name: {plasmid_name}')
                download_plasmid_sequence(plasmid_id, save_dir)
    else:
        print(f'Failed to access search results URL: {search_url}')

# Main function to scrape all plasmid sequences from the search results
def main():
    base_url = 'https://www.addgene.org/search/catalog/plasmids/?q=&page_size=20&expression=Bacterial+Expression'
    
    # Process the first page
    print("Starting to process the first page.")
    process_search_results(base_url)

    # Process subsequent pages if needed
    total_pages = 1354  # Total number of pages
    for page in range(2, total_pages + 1):
        paginated_url = f'{base_url}&page={page}'
        print(f"Processing page {page}.")
        process_search_results(paginated_url)

if __name__ == '__main__':
    main()
