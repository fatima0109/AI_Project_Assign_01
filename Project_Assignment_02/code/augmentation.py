"""
Simplified Data Augmentation
"""
import random
import pandas as pd
import numpy as np

class SimpleAugmenter:
    def __init__(self):
        self.synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'question': ['query', 'inquiry'],
            'answer': ['response', 'reply'],
            'evade': ['avoid', 'dodge'],
            'avoid': ['evade', 'elude']
        }
    
    def synonym_replace(self, text, p=0.2):
        """Simple synonym replacement."""
        words = text.split()
        if len(words) < 3:
            return text
        
        n_replace = max(1, int(len(words) * p))
        
        for _ in range(n_replace):
            idx = random.randint(0, len(words)-1)
            word = words[idx].lower().strip('.,!?;:')
            
            if word in self.synonyms:
                replacement = random.choice(self.synonyms[word])
                words[idx] = replacement
        
        return ' '.join(words)
    
    def augment_dataset(self, X, y, target_class=None, n_samples=200):
        """Augment minority class."""
        if target_class is not None:
            minority_mask = (y == target_class)
            minority_texts = X[minority_mask].tolist()
        else:
            # Find minority class
            class_counts = y.value_counts()
            minority_class = class_counts.idxmin()
            minority_mask = (y == minority_class)
            minority_texts = X[minority_mask].tolist()
        
        augmented_texts = []
        augmented_labels = []
        
        for _ in range(n_samples):
            text = random.choice(minority_texts)
            aug_text = self.synonym_replace(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(target_class if target_class else minority_class)
        
        # Combine
        X_aug = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
        y_aug = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
        
        print(f"Added {n_samples} augmented samples")
        
        return X_aug, y_aug