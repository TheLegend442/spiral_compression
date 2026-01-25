
import torch
import numpy as np
from chronos import Chronos2Pipeline

class ChronosEmbedder:
    def __init__(self, model_name: str = "amazon/chronos-2"):
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map="auto",
        )

    def embed_batch(self, series_batch, debug=False):
        batch_size = len(series_batch)
        
        # Get embeddings without tracking gradients
        with torch.no_grad():
            embeddings_list, _ = self.pipeline.embed(
                series_batch,
                batch_size=batch_size,
            )
        
        if debug:
            print(type(embeddings_list))
            print(len(embeddings_list))
            print(embeddings_list[0].shape)
        
    
        # Extracts sequence embeddings + removes special tokens
        seq_embs = []
        for emb in embeddings_list:
            emb_patches = emb[:, 1:-1, :] 
            seq_emb = emb_patches.mean(axis=0).cpu().numpy()
            seq_embs.append(seq_emb)
            
        return seq_embs
    
    def embed_single(self, series):
        with torch.no_grad():
            embeddings_list, _ = self.pipeline.embed(
                [series],
                batch_size=1,
            )
        
        emb = embeddings_list[0]
        emb_patches = emb[:, 1:-1, :] 
        seq_emb = emb_patches.mean(axis=0).cpu().numpy()
        return seq_emb

def main(debug):
    # code for debugging
    
    model = ChronosEmbedder()
    series_batch = []
    
    for i in range(16):
        temp_r = np.random.rand(200).astype(np.float32)
        series_batch.append(temp_r)
    
    print("Length of a single spiral: ", len(series_batch[0]))
    seq_embs = model.embed_batch(series_batch, debug=debug)
    
    # print(f"Generated {len(seq_embs)} sequence embeddings.")
    print(len(series_batch[0])/seq_embs[0].shape[0])

if __name__ == '__main__':
    main(debug=True)