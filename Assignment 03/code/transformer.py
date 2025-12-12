"""
Improved transformer implementation without TensorFlow dependencies.
Uses PyTorch only for cleaner implementation.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np

class ImprovedTransformer(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_labels: int = 2, dropout_rate: float = 0.3):
        """
        Improved transformer with attention mechanisms.
        
        Args:
            model_name: Pretrained model name from HuggingFace
            num_labels: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Load configuration and base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # Additional attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Classification head with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            logits: Classification logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply additional attention
        attn_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Residual connection and layer norm
        sequence_output = self.layer_norm(sequence_output + attn_output)
        
        # Use [CLS] token for classification
        pooled_output = sequence_output[:, 0, :]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(self, tokenizer, texts, batch_size=16, device='cpu'):
        """
        Make predictions on a list of texts.
        
        Args:
            tokenizer: HuggingFace tokenizer
            texts: List of text strings
            batch_size: Batch size for prediction
            device: Device to run on ('cpu' or 'cuda')
            
        Returns:
            predictions: Array of predicted class indices
            probabilities: Array of class probabilities
        """
        self.eval()
        self.to(device)
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                logits = self(**inputs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)

def train_transformer_model(X_train, y_train, X_val, y_val, 
                          model_name="distilbert-base-uncased",
                          num_epochs=3, batch_size=16, learning_rate=2e-5,
                          output_dir="models/proposed_transformer"):
    """
    Train the improved transformer model.
    
    Returns:
        model: Trained model
        tokenizer: Trained tokenizer
        history: Training history
    """
    import os
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AdamW
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    num_labels = len(label_encoder.classes_)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ImprovedTransformer(model_name, num_labels=num_labels)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create Dataset class
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(X_train, y_train_encoded, tokenizer)
    val_dataset = TextDataset(X_val, y_val_encoded, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    print(f"\nModel saved to {output_dir}")
    
    return model, tokenizer, history, label_encoder