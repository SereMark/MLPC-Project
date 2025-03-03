# Audio ML Management Dashboard

This project provides a **Streamlit** app for managing basic audio machine learning workflows. It features three main steps: **Data Preparation**, **Training**, and **Evaluation**.

## Setup

1. **Install Conda**.  
2. **Clone the Repository**:  
   ```bash
   git clone <REPO_URL>
   cd <REPO_FOLDER>
   ```
3. **Create and Activate the Environment**:  
   ```bash
   conda env create -f environment.yml
   conda activate mlpc-project
   ```

## Usage

1. **Start the Dashboard**:  
   ```bash
   streamlit run src/dashboard.py
   ```
2. **Select a Tab**:  
   - **Data Preparation**: Specify raw audio/annotations directories and basic audio settings.  
   - **Training**: Configure epochs, batch size, learning rate, and data paths.  
   - **Evaluation**: Provide a trained model path, test data path, and output location for results.