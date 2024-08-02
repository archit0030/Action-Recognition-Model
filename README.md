# Action-Recognition-Model

## Video Classification Using VGG16 and BiLSTM

### Overview

This repository provides a script for video classification using a deep learning model that combines the power of VGG16 for feature extraction and BiLSTM for capturing temporal dependencies in video sequences. The workflow includes:

- **GPU Configuration**: Sets up GPU for TensorFlow.
- **Data Loading**: Loads preprocessed video data from a pickle file.
- **Model Definition**: Builds a neural network using VGG16 and BiLSTM.
- **Model Training**: Compiles and trains the model.
- **Model Evaluation**: Evaluates the trained model on test data.
- **Visualization**: Plots and saves training history graphs.

### Key Features

- **Pre-trained VGG16**: Efficient feature extraction.
- **Bi-LSTM Architecture**: Captures temporal dependencies in video sequences.
- **Layer Normalization**: Enhances model performance.

### Requirements

- Python 3.9
- TensorFlow
- NumPy
- Pickle
- OpenCV
- Matplotlib

### Installation ###

1. **Clone this repository:**

   ```bash
   git clone https://github.com/archit0030/Action-Recognition-Model.git

    

2.  **Create a virtual environment (recommended) and install the required dependencies:**
    ```Bash
    python -m venv env
    
    source env/bin/activate  # Activate on Linux/macOS
    
    pip install -r requirements.txt  # Assuming a requirements.txt file is present

***Data Preparation:***

    Prepare your video data and create training, validation, and test sets.
    
    Preprocess the data (e.g., resizing, normalization) and convert it into a format suitable for the model (e.g., NumPy arrays).
    
    Save the preprocessed data as a pickle file (your.pkl) in the same directory as the script.

***Usage:***
  
    Run the script video classification model to train and evaluate the model:
    Bash
    python video classification model.py
    Use code with caution.
    
    This script assumes the preprocessed data is saved as your.pkl.
    
    
    To train a custom pkl file run the code:
    Bash
    python video pickle_file.py
    
    You can modify the script to customize training parameters (epochs, batch size) and data paths.


***Explanation:***

    Import Libraries: The script imports necessary libraries for data manipulation, model building, training, visualization, and GPU configuration.
    
    Set GPU Configuration (Optional): This section attempts to configure GPU usage for memory efficiency. If you encounter issues, comment out this section.
    
    Load Data: The script loads preprocessed video data from the pickle file. Modify the file path if needed.

***Model Architecture:***

    A pre-trained VGG16 model is loaded with weights from ImageNet, but its top layers are frozen to prevent overfitting.
    
    TimeDistributed layers are used to apply VGG16 to each frame in a video sequence.
    
    GlobalAveragePooling2D is used to summarize features from each frame.
    
    Layer normalization is added for improved model stability.
    
    A Bi-LSTM layer captures temporal relationships between frames. You can experiment with different configurations (e.g., number of units, return sequences).
    
    Dropout layers help prevent overfitting.
    
    Dense layers with appropriate activation functions and regularization (L2) project the features to output probabilities for action classes.

    Model Compilation: The model is compiled with an Adam optimizer, sparse categorical cross-entropy loss (suitable for multi-class classification), and accuracy metric.
    
    Early Stopping (Optional): You can uncomment this section to implement early stopping to prevent overtraining.
    
    TensorBoard Callback: A TensorBoard callback is created to log training and validation metrics for visualization.
    
    Model Training: The model is trained using the fit method, specifying the training and validation data, epochs, batch size, and callback.
    
    Model Saving: The trained model is saved as a .h5 file.
    
    Model Evaluation: The model's performance is evaluated on the test data, and test loss and accuracy are printed.
    
    Training History Visualization: Training and validation accuracy/loss curves are plotted using Matplotlib and saved as images.

***Customization:***
    
    Modify the data loading section (data_file) to point to your preprocessed data file.
        
        Experiment with different Bi-LSTM configurations (units, return sequences) to potentially improve performance.
        
        Adjust regularization parameters (L2 weight) based on your dataset.
        
        Consider hyperparameter tuning using techniques like grid search or random search to find optimal settings.
