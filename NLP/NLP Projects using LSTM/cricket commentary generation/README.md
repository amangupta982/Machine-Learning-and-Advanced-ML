🏏 Cricket Commentary Generator using LSTM


📌 Project Overview

The Cricket Commentary Generator is a Neural Language Model built using Long Short-Term Memory (LSTM) networks. The system learns patterns from real cricket match commentary and generates realistic cricket commentary automatically based on a user-provided input prompt.

This project demonstrates the application of Deep Learning and Natural Language Processing (NLP) for automated text generation in the sports domain.


🎯 Objective

The objective of this project is to:
	•	Build a Neural Language Model using LSTM
	•	Learn sequential language patterns from cricket commentary data
	•	Predict the next word in a sequence
	•	Generate realistic cricket commentary automatically
	•	Demonstrate practical NLP text generation using deep learning


🚀 Features

✅ Neural Language Model using LSTM
✅ Bidirectional LSTM architecture
✅ Real-world cricket commentary dataset
✅ Automatic text generation
✅ Top-K sampling for diverse output
✅ Temperature-based prediction
✅ Model saving and loading (no retraining required)
✅ Memory-efficient training using sparse categorical loss
✅ Interactive terminal-based commentary generation


📂 Dataset

The dataset consists of ball-by-ball cricket commentary stored in multiple CSV files.

Dataset structure:
COMMENTARY_INTL_MATCH/
 ├── 1122886_COMMENTARY.csv
 ├── 1122887_COMMENTARY.csv
 ├── ...

 Each CSV file contains a column named:
 Commentary

 which is used for training the language model.

 ⚙️ Technologies Used
	•	Python
	•	TensorFlow / Keras
	•	NumPy
	•	Pandas
	•	Natural Language Processing (NLP)
	•	LSTM (Long Short-Term Memory)


🏗️ Project Structure
Cricket-Commentary-Generator/
│
├── COMMENTARY_INTL_MATCH/
│   ├── *.csv
│
├── cricket_commentary_generator.py
├── cricket_model.h5
├── tokenizer.pkl
├── README.md


▶️ How to Run the Project

1️⃣ Install Dependencies
pip install tensorflow pandas numpy

2️⃣ Run the Program
python cricket_commentary_generator.py