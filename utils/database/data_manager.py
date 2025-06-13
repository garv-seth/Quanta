
"""
Database Data Manager for Financial Models

Handles storing and retrieving model weights, training data, and performance metrics
using Replit's built-in database system.
"""

import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from replit import db


class ModelDataManager:
    """Manages model data storage and retrieval using Replit database"""
    
    def __init__(self):
        self.db = db
        self.model_prefix = "financial_model_"
        self.training_data_prefix = "training_data_"
        self.performance_prefix = "performance_"
    
    def save_model_weights(self, model_name: str, model_data: Dict) -> str:
        """Save model weights and configuration to database"""
        try:
            # Prepare model data for storage
            storage_data = {
                'model_name': model_name,
                'config': model_data.get('config', {}),
                'tokenizer_vocab': model_data.get('tokenizer_vocab', {}),
                'tokenizer_inverse_vocab': model_data.get('tokenizer_inverse_vocab', {}),
                'training_history': model_data.get('training_history', []),
                'model_info': model_data.get('model_info', {}),
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Convert numpy arrays to lists for JSON serialization
            if 'token_embeddings' in model_data:
                storage_data['token_embeddings'] = self._serialize_array(model_data['token_embeddings'])
            if 'position_embeddings' in model_data:
                storage_data['position_embeddings'] = self._serialize_array(model_data['position_embeddings'])
            if 'time_embeddings' in model_data:
                storage_data['time_embeddings'] = self._serialize_array(model_data['time_embeddings'])
            if 'output_projection' in model_data:
                storage_data['output_projection'] = self._serialize_array(model_data['output_projection'])
            if 'output_bias' in model_data:
                storage_data['output_bias'] = self._serialize_array(model_data['output_bias'])
            
            # Store in database
            key = f"{self.model_prefix}{model_name}"
            self.db[key] = storage_data
            
            # Store model metadata
            self._update_model_registry(model_name, storage_data)
            
            return key
            
        except Exception as e:
            raise Exception(f"Failed to save model weights: {str(e)}")
    
    def load_model_weights(self, model_name: str) -> Dict:
        """Load model weights and configuration from database"""
        try:
            key = f"{self.model_prefix}{model_name}"
            
            if key not in self.db:
                raise ValueError(f"Model '{model_name}' not found in database")
            
            storage_data = self.db[key]
            
            # Deserialize arrays
            model_data = {
                'config': storage_data.get('config', {}),
                'tokenizer_vocab': storage_data.get('tokenizer_vocab', {}),
                'tokenizer_inverse_vocab': storage_data.get('tokenizer_inverse_vocab', {}),
                'training_history': storage_data.get('training_history', []),
                'model_info': storage_data.get('model_info', {}),
                'saved_at': storage_data.get('saved_at'),
                'version': storage_data.get('version', '1.0')
            }
            
            # Deserialize numpy arrays
            if 'token_embeddings' in storage_data:
                model_data['token_embeddings'] = self._deserialize_array(storage_data['token_embeddings'])
            if 'position_embeddings' in storage_data:
                model_data['position_embeddings'] = self._deserialize_array(storage_data['position_embeddings'])
            if 'time_embeddings' in storage_data:
                model_data['time_embeddings'] = self._deserialize_array(storage_data['time_embeddings'])
            if 'output_projection' in storage_data:
                model_data['output_projection'] = self._deserialize_array(storage_data['output_projection'])
            if 'output_bias' in storage_data:
                model_data['output_bias'] = self._deserialize_array(storage_data['output_bias'])
            
            return model_data
            
        except Exception as e:
            raise Exception(f"Failed to load model weights: {str(e)}")
    
    def save_training_data(self, dataset_name: str, texts: List[str], metadata: Dict = None) -> str:
        """Save training data to database"""
        try:
            training_data = {
                'dataset_name': dataset_name,
                'texts': texts,
                'metadata': metadata or {},
                'num_texts': len(texts),
                'saved_at': datetime.now().isoformat(),
                'total_characters': sum(len(text) for text in texts),
                'avg_length': np.mean([len(text.split()) for text in texts])
            }
            
            key = f"{self.training_data_prefix}{dataset_name}"
            self.db[key] = training_data
            
            return key
            
        except Exception as e:
            raise Exception(f"Failed to save training data: {str(e)}")
    
    def load_training_data(self, dataset_name: str) -> List[str]:
        """Load training data from database"""
        try:
            key = f"{self.training_data_prefix}{dataset_name}"
            
            if key not in self.db:
                raise ValueError(f"Training data '{dataset_name}' not found in database")
            
            training_data = self.db[key]
            return training_data.get('texts', [])
            
        except Exception as e:
            raise Exception(f"Failed to load training data: {str(e)}")
    
    def save_performance_metrics(self, model_name: str, metrics: Dict) -> str:
        """Save model performance metrics"""
        try:
            performance_data = {
                'model_name': model_name,
                'metrics': metrics,
                'recorded_at': datetime.now().isoformat()
            }
            
            key = f"{self.performance_prefix}{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.db[key] = performance_data
            
            return key
            
        except Exception as e:
            raise Exception(f"Failed to save performance metrics: {str(e)}")
    
    def get_model_list(self) -> List[Dict]:
        """Get list of all stored models"""
        try:
            models = []
            
            # Get model registry
            if 'model_registry' in self.db:
                registry = self.db['model_registry']
                for model_name, info in registry.items():
                    models.append({
                        'name': model_name,
                        'saved_at': info.get('saved_at'),
                        'config': info.get('config', {}),
                        'training_epochs': len(info.get('training_history', []))
                    })
            
            return models
            
        except Exception as e:
            return []
    
    def get_training_datasets(self) -> List[Dict]:
        """Get list of all stored training datasets"""
        try:
            datasets = []
            
            for key in self.db.keys():
                if key.startswith(self.training_data_prefix):
                    dataset_data = self.db[key]
                    datasets.append({
                        'name': dataset_data.get('dataset_name'),
                        'num_texts': dataset_data.get('num_texts', 0),
                        'total_characters': dataset_data.get('total_characters', 0),
                        'avg_length': dataset_data.get('avg_length', 0),
                        'saved_at': dataset_data.get('saved_at')
                    })
            
            return datasets
            
        except Exception as e:
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from database"""
        try:
            key = f"{self.model_prefix}{model_name}"
            
            if key in self.db:
                del self.db[key]
                
                # Update registry
                if 'model_registry' in self.db:
                    registry = self.db['model_registry']
                    if model_name in registry:
                        del registry[model_name]
                        self.db['model_registry'] = registry
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def delete_training_data(self, dataset_name: str) -> bool:
        """Delete training data from database"""
        try:
            key = f"{self.training_data_prefix}{dataset_name}"
            
            if key in self.db:
                del self.db[key]
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database usage statistics"""
        try:
            total_keys = len(list(self.db.keys()))
            model_count = len([k for k in self.db.keys() if k.startswith(self.model_prefix)])
            dataset_count = len([k for k in self.db.keys() if k.startswith(self.training_data_prefix)])
            performance_count = len([k for k in self.db.keys() if k.startswith(self.performance_prefix)])
            
            return {
                'total_keys': total_keys,
                'model_count': model_count,
                'dataset_count': dataset_count,
                'performance_records': performance_count,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'total_keys': 0,
                'model_count': 0,
                'dataset_count': 0,
                'performance_records': 0,
                'error': str(e)
            }
    
    def _serialize_array(self, array: np.ndarray) -> List:
        """Serialize numpy array for JSON storage"""
        if isinstance(array, np.ndarray):
            return {
                'data': array.tolist(),
                'shape': array.shape,
                'dtype': str(array.dtype)
            }
        return array
    
    def _deserialize_array(self, array_data) -> np.ndarray:
        """Deserialize array from JSON storage"""
        if isinstance(array_data, dict) and 'data' in array_data:
            return np.array(array_data['data'], dtype=array_data.get('dtype', 'float32')).reshape(array_data['shape'])
        return np.array(array_data)
    
    def _update_model_registry(self, model_name: str, model_data: Dict):
        """Update the model registry with new model info"""
        try:
            if 'model_registry' not in self.db:
                self.db['model_registry'] = {}
            
            registry = self.db['model_registry']
            registry[model_name] = {
                'saved_at': model_data.get('saved_at'),
                'config': model_data.get('config', {}),
                'training_history': model_data.get('training_history', []),
                'version': model_data.get('version', '1.0')
            }
            
            self.db['model_registry'] = registry
            
        except Exception as e:
            pass  # Registry update is optional
    
    def backup_database(self) -> Dict:
        """Create a backup of all database content"""
        try:
            backup_data = {}
            
            for key in self.db.keys():
                backup_data[key] = self.db[key]
            
            backup_info = {
                'backup_created_at': datetime.now().isoformat(),
                'total_keys': len(backup_data),
                'data': backup_data
            }
            
            return backup_info
            
        except Exception as e:
            raise Exception(f"Failed to create backup: {str(e)}")
    
    def restore_database(self, backup_data: Dict) -> bool:
        """Restore database from backup"""
        try:
            if 'data' not in backup_data:
                raise ValueError("Invalid backup format")
            
            # Clear existing data (optional)
            # for key in list(self.db.keys()):
            #     del self.db[key]
            
            # Restore data
            for key, value in backup_data['data'].items():
                self.db[key] = value
            
            return True
            
        except Exception as e:
            return False


# Singleton instance
model_data_manager = ModelDataManager()
