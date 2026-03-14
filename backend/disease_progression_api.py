"""
Flask API Integration for Disease Progression Detection

Endpoints:
- POST /api/disease/upload-day: Upload image for a specific day
- POST /api/disease/analyze-progression: Analyze complete sequence
- GET /api/disease/status: Check upload status
"""

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
# from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Import model utilities
# import sys
# sys.path.append('../data_ml/notebooks/disease_progression')
# from disease_progression_model import GradCAMVisualizer


class DiseaseProgressionAPI:
    """
    API handler for disease progression detection
    """
    
    def __init__(self, model_path: str, class_names_path: str):
        """
        Initialize API with trained model
        
        Args:
            model_path: Path to saved model
            class_names_path: Path to class names file
        """
        from tensorflow import keras
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.class_names = np.load(class_names_path, allow_pickle=True)
        self.image_size = (224, 224)
        
        # Storage for multi-day uploads
        self.user_sequences = {}  # user_id -> {day: image}

        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {len(self.class_names)}")
    
    def upload_day_image(
        self,
        user_id: str,
        day: int,
        image_data: str
    ) -> Dict:
        """
        Upload image for a specific day
        
        Args:
            user_id: Unique user identifier
            day: Day number (1-5)
            image_data: Base64 encoded image
        
        Returns:
            Status response
        """
        try:
            # Decode image
            image = self._decode_base64_image(image_data)
            
            # Initialize user sequence if needed
            if user_id not in self.user_sequences:
                self.user_sequences[user_id] = {
                    'images': {},
                    'created_at': datetime.now().isoformat()
                }
            
            # Store image
            self.user_sequences[user_id]['images'][day] = image
            self.user_sequences[user_id]['updated_at'] = datetime.now().isoformat()
            
            # Check if sequence is complete
            days_uploaded = sorted(self.user_sequences[user_id]['images'].keys())
            is_complete = len(days_uploaded) >= 3  # Minimum 3 days
            
            return {
                'success': True,
                'user_id': user_id,
                'day': day,
                'days_uploaded': days_uploaded,
                'is_complete': is_complete,
                'message': f'Image for day {day} uploaded successfully'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_progression(self, user_id: str) -> Dict:
        """
        Analyze disease progression from uploaded sequence
        
        Args:
            user_id: User identifier
        
        Returns:
            Analysis results
        """
        try:
            if user_id not in self.user_sequences:
                return {
                    'success': False,
                    'error': 'No images found for this user'
                }
            
            user_data = self.user_sequences[user_id]
            images = user_data['images']
            
            if len(images) < 3:
                return {
                    'success': False,
                    'error': f'Minimum 3 images required, only {len(images)} uploaded'
                }
            
            # Prepare sequence
            sequence = self._prepare_sequence(images)
            
            # Make prediction
            predictions = self.model.predict(np.expand_dims(sequence, 0), verbose=0)
            
            # Parse results
            disease_probs = predictions['disease_classification'][0]
            disease_idx = np.argmax(disease_probs)
            disease_name = self.class_names[disease_idx]
            confidence = float(disease_probs[disease_idx])
            
            severity_score = float(predictions['severity_score'][0][0])
            progression_rate = float(predictions['progression_rate'][0][0])
            
            # Calculate severity timeline
            severity_timeline = self._calculate_severity_timeline(
                severity_score,
                progression_rate,
                len(images)
            )
            
            # Generate recommendations
            recommendation = self._generate_recommendation(
                disease_name,
                severity_score,
                progression_rate
            )
            
            # Get top 3 predictions
            top_3_indices = np.argsort(disease_probs)[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': self.class_names[idx],
                    'confidence': float(disease_probs[idx])
                }
                for idx in top_3_indices
            ]
            
            # Prepare response
            response = {
                'success': True,
                'user_id': user_id,
                'analysis': {
                    'disease': disease_name,
                    'confidence': confidence,
                    'severity_score': severity_score,
                    'severity_level': self._get_severity_level(severity_score),
                    'progression_rate': progression_rate,
                    'progression_status': self._get_progression_status(progression_rate),
                    'severity_timeline': severity_timeline,
                    'days_analyzed': sorted(images.keys()),
                    'top_predictions': top_3_predictions
                },
                'recommendation': recommendation,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Clean up user data
            del self.user_sequences[user_id]
            
            return response
        
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def get_status(self, user_id: str) -> Dict:
        """
        Get upload status for user
        
        Args:
            user_id: User identifier
        
        Returns:
            Status information
        """
        if user_id not in self.user_sequences:
            return {
                'user_id': user_id,
                'days_uploaded': [],
                'is_complete': False,
                'message': 'No images uploaded yet'
            }
        
        user_data = self.user_sequences[user_id]
        days_uploaded = sorted(user_data['images'].keys())
        
        return {
            'user_id': user_id,
            'days_uploaded': days_uploaded,
            'is_complete': len(days_uploaded) >= 3,
            'created_at': user_data.get('created_at'),
            'updated_at': user_data.get('updated_at'),
            'message': f'{len(days_uploaded)} images uploaded'
        }
    
    def _decode_base64_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Resize
        image = image.resize(self.image_size)
        
        # Convert to numpy
        return np.array(image)
    
    def _prepare_sequence(self, images: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Prepare image sequence for model input
        
        Args:
            images: Dictionary of day -> image
        
        Returns:
            Sequence array (seq_len, H, W, 3)
        """
        # Sort by day
        sorted_days = sorted(images.keys())
        
        # Get images in order
        sequence = [images[day] for day in sorted_days]
        
        # Pad if needed (to 5 images)
        while len(sequence) < 5:
            # Repeat last image
            sequence.append(sequence[-1])
        
        # Take first 5 if more than 5
        sequence = sequence[:5]
        
        # Normalize
        sequence = np.array(sequence, dtype=np.float32) / 255.0
        
        return sequence
    
    def _calculate_severity_timeline(
        self,
        final_severity: float,
        progression_rate: float,
        num_days: int
    ) -> List[float]:
        """Calculate severity for each day"""
        timeline = []
        for day in range(num_days):
            severity = max(0.0, final_severity - progression_rate * (num_days - 1 - day))
            timeline.append(round(severity, 3))
        return timeline
    
    def _get_severity_level(self, severity: float) -> str:
        """Convert severity score to level"""
        if severity < 0.2:
            return 'Healthy'
        elif severity < 0.4:
            return 'Mild'
        elif severity < 0.6:
            return 'Moderate'
        elif severity < 0.8:
            return 'Severe'
        else:
            return 'Critical'
    
    def _get_progression_status(self, rate: float) -> str:
        """Get progression status from rate"""
        if rate < 0.05:
            return 'Stable'
        elif rate < 0.15:
            return 'Slowly Worsening'
        elif rate < 0.25:
            return 'Moderately Worsening'
        else:
            return 'Rapidly Worsening'
    
    def _generate_recommendation(
        self,
        disease: str,
        severity: float,
        progression_rate: float
    ) -> Dict:
        """Generate treatment recommendations"""
        urgency = 'low'
        if severity > 0.7 or progression_rate > 0.2:
            urgency = 'high'
        elif severity > 0.4 or progression_rate > 0.1:
            urgency = 'medium'
        
        # Generic recommendations (can be expanded with disease-specific advice)
        recommendations = {
            'urgency': urgency,
            'actions': [],
            'treatments': [],
            'monitoring': []
        }
        
        if urgency == 'high':
            recommendations['actions'].append('Immediate intervention required')
            recommendations['actions'].append('Consult agricultural expert')
            recommendations['treatments'].append('Apply appropriate fungicide/pesticide')
            recommendations['monitoring'].append('Monitor daily')
        elif urgency == 'medium':
            recommendations['actions'].append('Take action within 1-2 days')
            recommendations['treatments'].append('Apply preventive treatments')
            recommendations['monitoring'].append('Monitor every 2 days')
        else:
            recommendations['actions'].append('Continue monitoring')
            recommendations['treatments'].append('Maintain good agricultural practices')
            recommendations['monitoring'].append('Monitor weekly')
        
        # Add disease-specific advice
        if 'blight' in disease.lower():
            recommendations['treatments'].append('Remove affected leaves')
            recommendations['treatments'].append('Improve air circulation')
        elif 'rust' in disease.lower():
            recommendations['treatments'].append('Apply sulfur-based fungicide')
        elif 'spot' in disease.lower():
            recommendations['treatments'].append('Avoid overhead watering')
        
        return recommendations


# Flask app integration
def create_disease_progression_routes(app: Flask, api: DiseaseProgressionAPI):
    """
    Add disease progression routes to Flask app
    
    Args:
        app: Flask application
        api: DiseaseProgressionAPI instance
    """
    
    @app.route('/api/disease/upload-day', methods=['POST'])
    def upload_day():
        """Upload image for a specific day"""
        try:
            data = request.json
            user_id = data.get('user_id')
            day = data.get('day')
            image_data = data.get('image')
            
            if not all([user_id, day, image_data]):
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields: user_id, day, image'
                }), 400
            
            result = api.upload_day_image(user_id, day, image_data)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/disease/analyze-progression', methods=['POST'])
    def analyze_progression():
        """Analyze disease progression"""
        try:
            data = request.json
            user_id = data.get('user_id')
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing user_id'
                }), 400
            
            result = api.analyze_progression(user_id)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/disease/status', methods=['GET'])
    def get_status():
        """Get upload status"""
        try:
            user_id = request.args.get('user_id')
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing user_id'
                }), 400
            
            result = api.get_status(user_id)
            return jsonify(result), 200
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# Example usage in main app.py
if __name__ == '__main__':
    from flask import Flask
    
    app = Flask(__name__)
    
    # Initialize API
    api = DiseaseProgressionAPI(
        model_path='../../models/disease_progression_final.h5',
        class_names_path='../../models/class_names.npy'
    )
    
    # Add routes
    create_disease_progression_routes(app, api)
    
    # Run app
    app.run(debug=True, port=5000)
