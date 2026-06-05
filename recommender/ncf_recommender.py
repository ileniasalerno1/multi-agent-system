import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

# --- SETUP PERCORSI ---
# uso os.path per rendere il codice portabile tra windows/linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'book_rich_metadata.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'data', 'models', 'ncf_model64.pth')

class NCF(nn.Module):
    """
    Architettura Neural Collaborative Filtering.
    Combina user e item embedding attraverso un MLP (Multi-Layer Perceptron).
    """
    def __init__(self, num_users, num_books, embedding_dim, hidden_layers, dropout=0.2):
        super(NCF, self).__init__()

        # Layer di Embedding: convertono ID in vettori densi
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)

        # Costruzione dinamica dei layer MLP
        layers = []
        input_dim = embedding_dim * 2   # concateniamo i due embedding

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Layer finale: produce un unico score di rilevanza
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_input, book_input):
        u_emb = self.user_embedding(user_input)
        b_emb = self.book_embedding(book_input)
        # uniamo i vettori e passiamo per la rete neurale
        x = torch.cat([u_emb, b_emb], dim=-1)
        out = self.mlp(x)
        return out.squeeze()

class NCFRecommender:
    """
    Wrapper per gestire il caricamento del modello e l'inferenza (Warm Start).
    """
    def __init__(self):
        print("Inizializzazione NCF Recommender (Warm Start)...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_users = 0
        self.num_books = 0
        
        # carico i metadati per arricchire le raccomandazioni con titoli e autori
        if os.path.exists(METADATA_PATH):
            self.books_df = pd.read_csv(METADATA_PATH)
            if 'book_id' in self.books_df.columns:
                self.books_df['book_id'] = self.books_df['book_id'].astype(int)
            
            # mapping veloce book_id -> record per non dover filtrare il DF ogni volta
            self.books_map = self.books_df.set_index('book_id').to_dict('index')
        else:
            print(f"AVVISO: Metadati non trovati in {METADATA_PATH}")
            self.books_map = {}

        # Caricamento del modello salvato
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                
                # estraggo le dimensioni dai pesi salvati
                u_weight = state_dict.get('user_embedding.weight')
                b_weight = state_dict.get('book_embedding.weight')
                
                if u_weight is not None and b_weight is not None:
                    self.num_users, emb_dim = u_weight.shape
                    self.num_books = b_weight.shape[0]
                                        
                    # hidden_layers fissi come da addestramento (128 -> 64)
                    self.model = NCF(
                        num_users=self.num_users, 
                        num_books=self.num_books, 
                        embedding_dim=emb_dim, 
                        hidden_layers=[128, 64]
                    )
                    
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device).eval()
                    print(f"Modello NCF pronto ({self.num_users} utenti, {self.num_books} libri).")
                else:
                    print("ERRORE: state_dict incompleto.")
            except Exception as e:
                print(f"Errore caricamento NCF: {e}")
        else:
            print(f"File modello {MODEL_PATH} non trovato. NCF disabilitato.")

    def is_user_known(self, user_id: int) -> bool:
        """Controlla se l'utente esiste nel dataset di addestramento."""
        if self.model is None: return False
        return 0 <= user_id < self.num_users

    def recommend(self, user_id: int, top_k=5):
        """Genera una lista di libri consigliandi calcolando lo score su tutto il catalogo."""
        if self.model is None: return []

        # Preparo i tensori per l'inferenza massiva su tutti i libri
        all_book_ids = torch.arange(self.num_books, dtype=torch.long).to(self.device)
        u_input = torch.tensor([user_id] * self.num_books, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            preds = self.model(u_input, all_book_ids)
        
        # Estraggo i top_k risultati migliori
        scores, indices = torch.topk(preds, top_k)
        
        indices = indices.cpu().numpy()
        scores = scores.cpu().numpy()
        
        recs = []
        for i, idx in enumerate(indices):
            b_id = int(idx)
            info = self.books_map.get(b_id, {})
            
            recs.append({
                "book_id": b_id,
                "title": info.get('title', 'Unknown Title'),
                "author": info.get('clean_author', 'Unknown'),
                "year": int(info.get('year', 0)),
                "category": info.get('simple_category', 'Unknown'),
                "description": str(info.get('description', ''))[:200] + "...",
                "score": round(float(scores[i]), 4),
                "llm_reason": "Based on your reading history (Neural Model)"
            })
            
        return recs
