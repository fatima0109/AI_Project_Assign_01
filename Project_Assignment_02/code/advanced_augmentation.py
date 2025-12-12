"""
Simplified but effective data augmentation techniques for text.
Doesn't require external dependencies.
"""
import random
import re
from typing import List, Tuple
import pandas as pd
import numpy as np

class SimpleAugmenter:
    def __init__(self, augment_methods: List[str] = ['synonym', 'random_swap', 'random_delete']):
        self.methods = augment_methods
        
        # Simple synonym dictionary
        self.synonyms = {
            'good': ['great', 'excellent', 'fine', 'superior'],
            'bad': ['poor', 'terrible', 'awful', 'subpar'],
            'question': ['query', 'inquiry', 'interrogation'],
            'answer': ['response', 'reply', 'solution'],
            'help': ['assist', 'aid', 'support'],
            'important': ['crucial', 'vital', 'significant'],
            'different': ['distinct', 'dissimilar', 'varying'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'miniature'],
            'fast': ['quick', 'rapid', 'speedy'],
            'slow': ['sluggish', 'leisurely', 'gradual'],
            'happy': ['joyful', 'content', 'pleased'],
            'sad': ['unhappy', 'depressed', 'sorrowful']
        }
    
    def synonym_replacement(self, text: str, p: float = 0.2) -> str:
        """Replace words with simple synonyms from dictionary"""
        words = text.split()
        n_replace = max(1, int(len(words) * p))
        
        for _ in range(n_replace):
            for i in range(len(words)):
                if random.random() < p/2:  # Probability to replace this word
                    word_lower = words[i].lower()
                    # Remove punctuation for matching
                    word_clean = re.sub(r'[^\w\s]', '', word_lower)
                    
                    if word_clean in self.synonyms:
                        synonym = random.choice(self.synonyms[word_clean])
                        # Preserve original capitalization
                        if words[i][0].isupper():
                            synonym = synonym.capitalize()
                        words[i] = synonym
                        break
        
        return ' '.join(words)
    
    def random_swap(self, text: str, p: float = 0.1) -> str:
        """Randomly swap two words in the sentence"""
        words = text.split()
        if len(words) >= 2:
            n_swap = max(1, int(len(words) * p))
            for _ in range(n_swap):
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words from the sentence"""
        words = text.split()
        if len(words) > 1:
            # Keep at least 2 words
            n_delete = max(1, min(int(len(words) * p), len(words) - 2))
            indices_to_delete = random.sample(range(len(words)), n_delete)
            indices_to_delete.sort(reverse=True)
            
            for idx in indices_to_delete:
                del words[idx]
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, p: float = 0.1) -> str:
        """Insert random words from the same sentence"""
        words = text.split()
        if words:
            n_insert = max(1, int(len(words) * p))
            for _ in range(n_insert):
                random_word = random.choice(words)
                position = random.randint(0, len(words))
                words.insert(position, random_word)
        
        return ' '.join(words)
    
    def back_translation_simulated(self, text: str) -> str:
        """Simulate back translation with simple word replacements"""
        words = text.split()
        modified_words = []
        
        for word in words:
            word_lower = word.lower()
            word_clean = re.sub(r'[^\w\s]', '', word_lower)
            
            # Simulate "translation" by sometimes replacing with synonyms
            if word_clean in self.synonyms and random.random() < 0.3:
                synonym = random.choice(self.synonyms[word_clean])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                modified_words.append(synonym)
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def augment_text(self, text: str) -> str:
        """Apply one random augmentation method"""
        if not self.methods:
            return text
        
        method = random.choice(self.methods)
        
        if method == 'synonym':
            return self.synonym_replacement(text)
        elif method == 'random_swap':
            return self.random_swap(text)
        elif method == 'random_delete':
            return self.random_deletion(text)
        elif method == 'random_insert':
            return self.random_insertion(text)
        elif method == 'backtranslation':
            return self.back_translation_simulated(text)
        else:
            return text
    
    def augment_dataset(self, X: pd.Series, y: pd.Series, 
                       target_class: int = None, n_samples: int = 500) -> Tuple[pd.Series, pd.Series]:
        """Augment dataset with focus on minority class"""
        if target_class is not None:
            minority_texts = X[y == target_class].tolist()
            minority_labels = y[y == target_class].tolist()
        else:
            minority_texts = X.tolist()
            minority_labels = y.tolist()
        
        if not minority_texts:
            return X, y
        
        augmented_texts = []
        augmented_labels = []
        
        for _ in range(n_samples):
            text = random.choice(minority_texts)
            label = random.choice(minority_labels)
            
            aug_text = self.augment_text(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
        
        # Combine with original
        all_texts = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
        all_labels = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
        
        return all_texts, all_labels
    
    def augment_dataset_balanced(self, X: pd.Series, y: pd.Series, 
                               target_size: int = None) -> Tuple[pd.Series, pd.Series]:
        """Augment all classes to reach target size or balance"""
        unique_classes = y.unique()
        class_counts = y.value_counts()
        
        if target_size is None:
            target_size = class_counts.max()
        
        augmented_texts = []
        augmented_labels = []
        
        for class_label in unique_classes:
            class_texts = X[y == class_label].tolist()
            current_count = len(class_texts)
            
            if current_count < target_size:
                needed = target_size - current_count
                for _ in range(needed):
                    text = random.choice(class_texts)
                    aug_text = self.augment_text(text)
                    augmented_texts.append(aug_text)
                    augmented_labels.append(class_label)
        
        # Combine with original
        all_texts = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
        all_labels = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
        
        return all_texts, all_labels