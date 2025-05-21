# Mancity Predictions for the 2023-24 Season

This project aims to predict the outcomes of Manchester City matches for the 2023-24 season using machine learning techniques. The project includes data analysis, preprocessing, feature engineering, model training, and evaluation.

## Project Structure

- **data/**: Contains the dataset used for training and testing the machine learning model.
  - `mancity23-24.csv`: Match data for Manchester City for the 2023-24 season.

- **notebooks/**: Contains Jupyter notebooks for data analysis and model prototyping.
  - `Mancity_predictions.ipynb`: Initial code for data analysis, preprocessing, and model training.

- **src/**: Contains the source code for the project.
  - `__init__.py`: Marks the `src` directory as a Python package.
  - `data_preprocessing.py`: Functions for loading and preprocessing the dataset.
  - `features.py`: Functions for feature engineering.
  - `model.py`: Implementation of the machine learning model.
  - `predict.py`: Functions for making predictions with the trained model.

- **tests/**: Contains unit tests for the project.
  - `__init__.py`: Marks the `tests` directory as a Python package.
  - `test_data_preprocessing.py`: Unit tests for data preprocessing functions.
  - `test_features.py`: Unit tests for feature engineering functions.
  - `test_model.py`: Unit tests for model training and evaluation functions.

- **requirements.txt**: Lists the dependencies required for the project, including libraries such as pandas and scikit-learn.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Mancity_predictions_for_season23-24
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebook for initial data analysis and model training:
   ```
   jupyter notebook notebooks/Mancity_predictions.ipynb
   ```

## Usage

- Use the `data_preprocessing.py` module to load and preprocess the dataset.
- Utilize the `features.py` module to extract relevant features for the model.
- Train the model using the `model.py` module and evaluate its performance.
- Make predictions using the `predict.py` module.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.