# Climate Risk Assessment: Wildfire Risk Classification

This project implements a deep learning-based approach for wildfire risk assessment using satellite imagery and geolocation data. The system combines a Variational Autoencoder (VAE) with Logistic Regression to classify wildfire risks in geographical areas.

## Project Structure

- `src/`
  - `models/` - Deep learning model implementations
    - `vae.py` - Variational Autoencoder implementation
    - `classifier.py` - Logistic Regression classifier
  - `data/` - Data processing and management
    - `data_loader.py` - Satellite image and geolocation data loading
    - `preprocessing.py` - Image preprocessing utilities
  - `utils/` - Utility functions
    - `visualization.py` - Visualization tools for results
    - `metrics.py` - Evaluation metrics
  - `config/` - Configuration files
    - `model_config.py` - Model hyperparameters
    - `data_config.py` - Data processing parameters
  - `main.py` - Main training and evaluation script

## Features

- Variational Autoencoder (VAE) for feature extraction from satellite imagery
- Logistic Regression classifier for wildfire risk assessment
- Integration of geolocation data with image features
- Automated risk classification pipeline
- Visualization tools for risk assessment results

## Usage

1. Prepare the data:
```bash
python src/data/prepare_data.py --input_dir /path/to/satellite/images --output_dir /path/to/processed/data
```

2. Train the models:
```bash
python src/main.py --config config/model_config.py --data_dir /path/to/processed/data
```

3. Evaluate and generate risk assessments:
```bash
python src/main.py --mode evaluate --model_path /path/to/trained/model --data_dir /path/to/test/data
```

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Matplotlib

## Implementation Details

The project implements a two-stage approach:

1. **Feature Extraction (VAE)**:
   - Encodes satellite images into a latent space
   - Captures important visual features for risk assessment
   - Handles image preprocessing and normalization

2. **Risk Classification (Logistic Regression)**:
   - Combines VAE features with geolocation data
   - Classifies areas into risk categories
   - Provides probability scores for risk assessment

## Data Requirements

- Satellite imagery of geographical areas
- Geolocation data (latitude/longitude)
- Historical wildfire data (for training)
- Risk assessment labels

## Model Architecture

The VAE-Logit architecture consists of:
1. VAE encoder for feature extraction
2. Latent space representation
3. Feature combination with geolocation data
4. Logistic Regression classifier
5. Risk probability output

## Evaluation Metrics

- Classification accuracy
- Precision and recall
- ROC-AUC scores
- Risk assessment confidence scores

## Future Work

- Integration with financial risk models
- Real-time risk assessment capabilities
- Multi-hazard risk assessment
- Integration with climate scenario analysis tools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

jhague@stanford.edu
