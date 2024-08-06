from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_path = '/home/stacys/data/nucleotide_transformer_embeddings.npy'
sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/all_sequences.npy', allow_pickle=True)
# sequences = [
#     "ATCGAATCGGCTAGCTAGCTAGCTAGCTAGCTAATCGATCGATCGATCGATCGATCGATCGATCGTCGATCG",
#     "GCTAGCTTAGCTAGCTAGCACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATGCTA",
#     "CGATCGCGATCGATCGATATCGATCCGATCGATCGATATGCGATCGATCCGATCCCGATCGATGATATCGAT"
# ]
plasmid_ids = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/plasmid_ids.npy')

def main():
    '''
    Main function:
    1) Loads the tokenizer and the model.
    2) Tokenizes the input sequences.
    3) Runs input sequences in batches for tokenization, attention mask, and model inference to obtain the embeddings.
    4) Saves the embeddings as a numpy file.
    '''

    print("Loading tokenizer and the model...")
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(device)

    max_length = tokenizer.model_max_length
    print(f"Max length: {max_length}")

    batch_size = 16
    
    def process_batches(sequences, batch_size):
        '''
        Processes sequences in batches.
        '''
        embeddings_list = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            try:
                print("Tokenization...")
                tokens = tokenizer.batch_encode_plus(
                    batch_sequences, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=max_length
                )
                tokens_ids = tokens["input_ids"].to(device)
                
                print("Creating attention mask...")
                attention_mask = (tokens_ids != tokenizer.pad_token_id).to(device)
                
                print("Model inference...")
                with torch.no_grad():
                    torch_outs = model(tokens_ids, attention_mask=attention_mask, output_hidden_states=True)
                    embeddings = torch_outs.hidden_states[-1].detach().cpu().numpy()
                
                print("Computing embeddings...")
                attention_mask = attention_mask.unsqueeze(-1).cpu().numpy()
                mean_sequence_embeddings = np.sum(attention_mask * embeddings, axis=-2) / np.sum(attention_mask, axis=1)
                
                embeddings_list.extend(mean_sequence_embeddings)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
        
        return np.array(embeddings_list)

    print("Processing sequences...")
    mean_sequence_embeddings = process_batches(sequences, batch_size)
    print(f"Mean sequence embeddings shape: {mean_sequence_embeddings.shape}")

    print(f"Saving the embeddings to {output_path}")
    np.save(output_path, mean_sequence_embeddings)

if __name__ == '__main__':
    main()