# poll-classifier

## Overview
This project classifies free response poll issues using a chat language model. The input is a CSV file where each row is a response. The language model outputs a list of issues coded following the provided codebook.

## Setup
1. Clone the repository.
2. Install the required dependencies.
3. Set up your Groq API key in the environment variables.

## Running the Classifier
1. Place your survey responses in a file named `responses.csv`.
2. Ensure the `CODEBOOK.txt` file is in the same directory.
3. Run the classifier script:
   ```bash
   python poll_classifier.py
   ```
4. The classified responses will be saved in `classified_responses.csv`.

## Files
- `poll_classifier.py`: Main script to classify survey responses.
- `classifier.py`: Helper functions or classes (currently empty).
- `CODEBOOK.txt`: Contains the codebook for classifying responses.