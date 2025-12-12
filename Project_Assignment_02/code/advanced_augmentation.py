"""
Simplified but effective data augmentation techniques for text.
Doesn't require external dependencies.
Enhanced for Assignment 3 with QEvasion dataset specific features.
"""
import random
import re
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

class SimpleAugmenter:
    def __init__(self, augment_methods: List[str] = ['synonym', 'random_swap', 'random_delete']):
        self.methods = augment_methods
        
        # Enhanced synonym dictionary for QEvasion dataset
        self.synonyms = {
            # Question/Answer related
            'question': ['query', 'inquiry', 'interrogation', 'quiz'],
            'answer': ['response', 'reply', 'solution', 'retort'],
            'ask': ['inquire', 'query', 'question'],
            'respond': ['answer', 'reply', 'retort'],
            
            # Evasion related
            'evade': ['avoid', 'dodge', 'elude', 'sidestep'],
            'avoid': ['evade', 'dodge', 'elude', 'circumvent'],
            'direct': ['straightforward', 'clear', 'explicit', 'unambiguous'],
            'indirect': ['evasive', 'vague', 'ambiguous', 'roundabout'],
            
            # General synonyms
            'good': ['great', 'excellent', 'fine', 'superior', 'outstanding'],
            'bad': ['poor', 'terrible', 'awful', 'subpar', 'inferior'],
            'help': ['assist', 'aid', 'support', 'facilitate'],
            'important': ['crucial', 'vital', 'significant', 'critical'],
            'different': ['distinct', 'dissimilar', 'varying', 'divergent'],
            'big': ['large', 'huge', 'enormous', 'substantial'],
            'small': ['tiny', 'little', 'miniature', 'minuscule'],
            'fast': ['quick', 'rapid', 'speedy', 'expeditious'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'deliberate'],
            'happy': ['joyful', 'content', 'pleased', 'delighted'],
            'sad': ['unhappy', 'depressed', 'sorrowful', 'melancholy'],
            
            # Technical terms
            'system': ['framework', 'structure', 'setup', 'configuration'],
            'model': ['framework', 'paradigm', 'architecture', 'design'],
            'data': ['information', 'facts', 'statistics', 'figures'],
            'algorithm': ['procedure', 'method', 'technique', 'process'],
            
            # Interview specific
            'interview': ['meeting', 'discussion', 'conversation', 'dialog'],
            'candidate': ['applicant', 'prospect', 'contender', 'aspirant'],
            'position': ['job', 'role', 'post', 'appointment'],
            'experience': ['background', 'history', 'expertise', 'knowledge'],
        }
        
        # QEvasion specific patterns
        self.qevasion_patterns = {
            'question_patterns': [
                r'what is your',
                r'can you tell me',
                r'how would you',
                r'why do you',
                r'describe your',
                r'tell me about',
                r'what are your',
                r'how do you'
            ],
            'evasive_patterns': [
                r'that\'s a good question',
                r'it depends on',
                r'that\'s difficult to answer',
                r'I would need more context',
                r'could you clarify',
                r'to be honest',
                r'frankly speaking',
                r'generally speaking'
            ]
        }
    
    def synonym_replacement(self, text: str, p: float = 0.2) -> str:
        """Replace words with simple synonyms from dictionary"""
        words = text.split()
        if len(words) == 0:
            return text
            
        n_replace = max(1, int(len(words) * p))
        
        for _ in range(n_replace):
            # Try to find a replaceable word
            replaceable_indices = []
            for i in range(len(words)):
                word_lower = words[i].lower()
                word_clean = re.sub(r'[^\w\s]', '', word_lower)
                if word_clean in self.synonyms:
                    replaceable_indices.append(i)
            
            if replaceable_indices:
                i = random.choice(replaceable_indices)
                word_lower = words[i].lower()
                word_clean = re.sub(r'[^\w\s]', '', word_lower)
                
                if word_clean in self.synonyms:
                    synonym = random.choice(self.synonyms[word_clean])
                    # Preserve original capitalization
                    if words[i][0].isupper():
                        synonym = synonym.capitalize()
                    # Preserve trailing punctuation
                    if words[i][-1] in '.,!?;:':
                        synonym += words[i][-1]
                    words[i] = synonym
        
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
                # Preserve punctuation
                if word[-1] in '.,!?;:':
                    synonym += word[-1]
                modified_words.append(synonym)
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def qevasion_specific_augmentation(self, text: str) -> str:
        """Apply QEvasion dataset specific augmentations"""
        words = text.split()
        
        # Check for question patterns
        for pattern in self.qevasion_patterns['question_patterns']:
            if re.search(pattern, text.lower()):
                # Sometimes rephrase the question
                if random.random() < 0.3:
                    rephrasings = [
                        "Could you elaborate on",
                        "What are your thoughts about",
                        "Please describe",
                        "Can you explain"
                    ]
                    # Simple rephrasing: replace the question start
                    text = random.choice(rephrasings) + text[text.find(' '):]
                break
        
        # Check for evasive patterns and sometimes make them more/less evasive
        for pattern in self.qevasion_patterns['evasive_patterns']:
            if re.search(pattern, text.lower()):
                # Sometimes make response more direct or more evasive
                if random.random() < 0.4:
                    if "that's a good question" in text.lower():
                        alternatives = [
                            "That's an interesting question",
                            "That's a complex question",
                            "That question requires careful consideration"
                        ]
                        text = text.replace("That's a good question", random.choice(alternatives))
                break
        
        return text
    
    def contextual_word_replacement(self, text: str, p: float = 0.15) -> str:
        """Replace words based on context (simple version)"""
        words = text.split()
        if len(words) < 3:
            return text
            
        # Find positions of words that have synonyms
        replaceable_positions = []
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w\s]', '', word.lower())
            if word_clean in self.synonyms and len(word_clean) > 2:
                replaceable_positions.append(i)
        
        if not replaceable_positions:
            return text
            
        # Replace some words
        n_replace = max(1, int(len(replaceable_positions) * p))
        positions_to_replace = random.sample(replaceable_positions, 
                                           min(n_replace, len(replaceable_positions)))
        
        for pos in positions_to_replace:
            word_clean = re.sub(r'[^\w\s]', '', words[pos].lower())
            if word_clean in self.synonyms:
                synonym = random.choice(self.synonyms[word_clean])
                # Preserve capitalization and punctuation
                if words[pos][0].isupper():
                    synonym = synonym.capitalize()
                if words[pos][-1] in '.,!?;:':
                    synonym += words[pos][-1]
                words[pos] = synonym
        
        return ' '.join(words)
    
    def augment_text(self, text: str, method: str = None) -> str:
        """Apply one random augmentation method or specified method"""
        if not text or not isinstance(text, str):
            return text
            
        if not self.methods:
            return text
        
        # Choose method if not specified
        if method is None:
            method = random.choice(self.methods)
        
        # Apply the chosen method
        if method == 'synonym':
            aug_text = self.synonym_replacement(text)
        elif method == 'random_swap':
            aug_text = self.random_swap(text)
        elif method == 'random_delete':
            aug_text = self.random_deletion(text)
        elif method == 'random_insert':
            aug_text = self.random_insertion(text)
        elif method == 'backtranslation':
            aug_text = self.back_translation_simulated(text)
        elif method == 'qevasion':
            aug_text = self.qevasion_specific_augmentation(text)
        elif method == 'contextual':
            aug_text = self.contextual_word_replacement(text)
        else:
            aug_text = text
        
        # Apply QEvasion specific augmentation with some probability
        if random.random() < 0.2:
            aug_text = self.qevasion_specific_augmentation(aug_text)
        
        return aug_text
    
    def augment_dataset(self, X: pd.Series, y: pd.Series, 
                       target_class: int = None, n_samples: int = 500,
                       method: str = None) -> Tuple[pd.Series, pd.Series]:
        """Augment dataset with focus on minority class"""
        if len(X) == 0 or len(y) == 0:
            return X, y
            
        if target_class is not None:
            minority_mask = (y == target_class)
            if minority_mask.sum() == 0:
                return X, y
            minority_texts = X[minority_mask].tolist()
            minority_labels = y[minority_mask].tolist()
        else:
            minority_texts = X.tolist()
            minority_labels = y.tolist()
        
        if not minority_texts:
            return X, y
        
        augmented_texts = []
        augmented_labels = []
        
        for _ in range(n_samples):
            idx = random.randint(0, len(minority_texts) - 1)
            text = minority_texts[idx]
            label = minority_labels[idx]
            
            # Ensure text is string
            if not isinstance(text, str):
                text = str(text)
            
            aug_text = self.augment_text(text, method)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
        
        # Combine with original
        if augmented_texts:
            all_texts = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
            all_labels = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
        else:
            all_texts = X
            all_labels = y
        
        return all_texts, all_labels
    
    def augment_dataset_balanced(self, X: pd.Series, y: pd.Series, 
                               target_size: int = None,
                               method: str = None) -> Tuple[pd.Series, pd.Series]:
        """Augment all classes to reach target size or balance"""
        if len(X) == 0 or len(y) == 0:
            return X, y
            
        unique_classes = np.unique(y)
        class_counts = y.value_counts()
        
        if target_size is None:
            target_size = class_counts.max()
        
        augmented_texts = []
        augmented_labels = []
        
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_texts = X[class_mask].tolist()
            current_count = class_mask.sum()
            
            if current_count < target_size:
                needed = target_size - current_count
                for _ in range(needed):
                    text = random.choice(class_texts)
                    # Ensure text is string
                    if not isinstance(text, str):
                        text = str(text)
                    
                    aug_text = self.augment_text(text, method)
                    augmented_texts.append(aug_text)
                    augmented_labels.append(class_label)
        
        # Combine with original
        if augmented_texts:
            all_texts = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
            all_labels = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
        else:
            all_texts = X
            all_labels = y
        
        return all_texts, all_labels
    
    def smart_augment(self, X: pd.Series, y: pd.Series, 
                     strategy: str = 'balanced',
                     n_samples_per_class: int = None) -> Tuple[pd.Series, pd.Series]:
        """
        Smart augmentation with different strategies
        
        Args:
            X: Text data
            y: Labels
            strategy: 'balanced', 'minority', or 'all'
            n_samples_per_class: Number of samples per class (if None, auto-calculated)
        
        Returns:
            Augmented text and labels
        """
        if strategy == 'balanced':
            return self.augment_dataset_balanced(X, y, target_size=None)
        
        elif strategy == 'minority':
            class_counts = y.value_counts()
            minority_class = class_counts.idxmin()
            if n_samples_per_class is None:
                n_samples_per_class = max(100, class_counts.max() - class_counts.min())
            return self.augment_dataset(X, y, target_class=minority_class, 
                                       n_samples=n_samples_per_class)
        
        elif strategy == 'all':
            # Augment all classes equally
            class_counts = y.value_counts()
            max_count = class_counts.max()
            if n_samples_per_class is None:
                n_samples_per_class = 200
            
            augmented_texts = []
            augmented_labels = []
            
            for class_label in class_counts.index:
                class_mask = (y == class_label)
                class_texts = X[class_mask].tolist()
                
                for _ in range(n_samples_per_class):
                    text = random.choice(class_texts)
                    if not isinstance(text, str):
                        text = str(text)
                    
                    aug_text = self.augment_text(text)
                    augmented_texts.append(aug_text)
                    augmented_labels.append(class_label)
            
            if augmented_texts:
                all_texts = pd.concat([X, pd.Series(augmented_texts)], ignore_index=True)
                all_labels = pd.concat([y, pd.Series(augmented_labels)], ignore_index=True)
                return all_texts, all_labels
            else:
                return X, y
        
        else:
            return X, y
    
    def analyze_augmentation_effect(self, original_X: pd.Series, original_y: pd.Series,
                                  augmented_X: pd.Series, augmented_y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the effect of augmentation
        
        Returns:
            Dictionary with augmentation statistics
        """
        analysis = {
            'original_samples': len(original_X),
            'augmented_samples': len(augmented_X),
            'added_samples': len(augmented_X) - len(original_X),
            'augmentation_ratio': (len(augmented_X) - len(original_X)) / len(original_X),
            
            'original_class_distribution': original_y.value_counts().to_dict(),
            'augmented_class_distribution': augmented_y.value_counts().to_dict(),
        }
        
        # Calculate class balance improvement
        original_counts = original_y.value_counts()
        augmented_counts = augmented_y.value_counts()
        
        if len(original_counts) > 1:
            original_imbalance = original_counts.max() / original_counts.min()
            augmented_imbalance = augmented_counts.max() / augmented_counts.min()
            analysis['original_imbalance_ratio'] = original_imbalance
            analysis['augmented_imbalance_ratio'] = augmented_imbalance
            analysis['imbalance_improvement'] = original_imbalance - augmented_imbalance
        
        # Text statistics
        if len(augmented_X) > 0:
            original_lengths = original_X.str.len()
            augmented_lengths = augmented_X.str.len()
            
            analysis['text_length_stats'] = {
                'original_mean': original_lengths.mean(),
                'original_std': original_lengths.std(),
                'augmented_mean': augmented_lengths.mean(),
                'augmented_std': augmented_lengths.std(),
                'length_change': augmented_lengths.mean() - original_lengths.mean()
            }
        
        return analysis
    
    def save_augmented_data(self, X: pd.Series, y: pd.Series,
                          output_path: str,
                          original_X: pd.Series = None,
                          original_y: pd.Series = None):
        """
        Save augmented data to file with metadata
        """
        df = pd.DataFrame({
            'text': X,
            'label': y,
            'is_augmented': False
        })
        
        # Mark augmented samples
        if original_X is not None and original_y is not None:
            original_set = set(zip(original_X.astype(str), original_y))
            augmented_mask = ~df.apply(lambda row: (str(row['text']), row['label']) in original_set, axis=1)
            df['is_augmented'] = augmented_mask
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Save analysis if original data provided
        if original_X is not None and original_y is not None:
            analysis = self.analyze_augmentation_effect(original_X, original_y, X, y)
            
            # Save analysis to JSON
            import json
            analysis_path = output_path.replace('.csv', '_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"✅ Augmented data saved to: {output_path}")
            print(f"✅ Analysis saved to: {analysis_path}")
            print(f"\n📊 Augmentation Statistics:")
            print(f"   Original samples: {analysis['original_samples']}")
            print(f"   Augmented samples: {analysis['augmented_samples']}")
            print(f"   Added samples: {analysis['added_samples']}")
            print(f"   Augmentation ratio: {analysis['augmentation_ratio']:.2%}")
            
            if 'imbalance_improvement' in analysis:
                print(f"   Imbalance improvement: {analysis['imbalance_improvement']:.2f}")
        
        return df


# Example usage function
def demonstrate_augmentation():
    """Demonstrate how to use the augmenter"""
    # Create sample data
    sample_texts = [
        "What is your greatest strength in technical interviews?",
        "How do you handle difficult questions about your experience?",
        "Can you describe a time when you had to solve a complex problem?",
        "That's a good question, I would need to think about that carefully.",
        "It depends on the context of the situation and the specific requirements."
    ]
    sample_labels = [0, 0, 0, 1, 1]  # 0 = direct, 1 = evasive
    
    X = pd.Series(sample_texts)
    y = pd.Series(sample_labels)
    
    # Create augmenter
    augmenter = SimpleAugmenter(['synonym', 'random_swap', 'qevasion'])
    
    print("Original data:")
    for text, label in zip(X, y):
        print(f"  Label {label}: {text}")
    
    print("\nAugmented samples:")
    for i in range(3):
        aug_text = augmenter.augment_text(sample_texts[i])
        print(f"  Original: {sample_texts[i]}")
        print(f"  Augmented: {aug_text}")
        print()
    
    # Balance the dataset
    X_aug, y_aug = augmenter.augment_dataset_balanced(X, y)
    print(f"Original size: {len(X)}, Augmented size: {len(X_aug)}")
    
    return augmenter, X_aug, y_aug


if __name__ == "__main__":
    # Run demonstration
    augmenter, X_aug, y_aug = demonstrate_augmentation()