# TrashNet Classifier

This repository contains a deep learning project to classify trash types using the TrashNet dataset. The project is developed using TensorFlow and organized into the following sections:

## Project Tree
trashnet-classifier/
├── notebooks/
│   └── model_development.ipynb
├── scripts/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── evaluation.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ci.yml


### Features
- Data preparation and preprocessing
- Exploratory data analysis
- Image classification using TensorFlow
- CI/CD pipeline for automated testing and deployment

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dinata16/trashnet-classifier.git
   cd trashnet-classifier 
   ```
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the notebook for development:
```
jupyter notebook notebooks/model_development.ipynb
```

## Usage
For data preparation:
```
python scripts/data_preparation.py
```

For model training:
```
python scripts/model_training.py
```

For evaluation:
```
python scripts/evaluation.py
```
