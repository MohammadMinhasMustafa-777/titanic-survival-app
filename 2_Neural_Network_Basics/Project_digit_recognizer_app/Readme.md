# Handwritten Digit Recognizer App

A simple Flask web app that allows users to draw a handwritten digit (0-9) on a canvas and get an AI prediction using a trained MNIST model.

## Features
- Draw digits on an HTML5 canvas (mouse or touch support)
- Real-time prediction with confidence score
- Clear button to reset canvas
- Built with Flask, TensorFlow/Keras, PIL

## Setup & Local Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python app.py`
3. Open: https://mohammadminhasmustafa-digit-recognizer-flask.hf.space/
4. Draw a digit → Predict

## Technologies
- Backend: Flask, TensorFlow/Keras
- Frontend: HTML5 Canvas, JavaScript
- Preprocessing: PIL for image handling

## Notes
- Model trained on MNIST dataset (>97% accuracy on test set)
- Best with clean, centered drawings — freehand may cause funny errors!

## License
MIT License