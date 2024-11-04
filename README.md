# RSNA-2024-lumbar-spine-degenerative-classification

# RSNA Spine Classification Project on Google Vertex AI

## Project Overview
Successfully developed and implemented a deep learning model for the RSNA 2024 Lumbar Spine Degenerative Disease Classification competition using Google Vertex AI platform.

## Technical Stack
- Platform: Google Vertex AI / Jupyter Notebook
- Framework: PyTorch
- Models: EfficientNet-B0 architecture with custom adaptations
- Libraries: MONAI, Albumentations, timm

## Key Achievements
- Implemented a 5-fold cross-validation strategy for robust model training
- Developed custom data preprocessing pipeline for medical imaging
- Created an efficient inference system for multi-class predictions
- Successfully processed spine MRI data for degenerative condition classification
- Achieved balanced probability distributions across different spinal conditions

## Technical Implementation Highlights
```python
# Sample code snippet showing model architecture
class SpineModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', 
                 pretrained=True, num_classes=25):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            in_chans=3,
            num_classes=0
        )
        n_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features, num_classes)
        )
```

## Results
- Successfully processed 25/25 cases with perfect probability distributions
- Implemented clinically relevant predictions for:
  - Canal Stenosis
  - Neural Foraminal Narrowing
  - Subarticular Stenosis
- Maintained competition-standard probability formats

## Links
- Competition Link: [RSNA 2024 Competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- GitHub Repository: [Your repo link]

## Skills Demonstrated
- Deep Learning
- Medical Image Analysis
- Python Programming
- Google Cloud Platform
- PyTorch
- Healthcare AI Applications
