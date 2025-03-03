#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from predict import predict
import importlib.util
import sys

class MyModel:
    """
    Model for next character prediction using RoBERTa.
    """

    @classmethod
    def load_training_data(cls):
        # Look for CSV in different locations
        data = []
        csv_paths = [
            'generated_sentences.csv',                  # Current directory
            '../generated_sentences.csv',               # Parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated_sentences.csv')  # Absolute path
        ]
        
        for csv_path in csv_paths:
            try:
                import csv
                print(f"Trying to load CSV from: {csv_path}")
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if row and len(row) > 0:
                            data.append(row[0])
                    
                print(f"Successfully loaded {len(data)} sentences from {csv_path}")
                break  # Exit the loop if successful
            except Exception as e:
                print(f"Could not load from {csv_path}")
        
        # If no CSV was loaded successfully, use fallback data
        if not data:
            print("Using fallback training sentences")
            data = [
                "Hello world. How are you today?",
                "The quick brown fox jumps over the lazy dog.",
                "I need to send a message to Earth immediately.",
                "Space station status report: all systems nominal.",
                "Mission control, we have a situation here.",
                "The astronauts are preparing for spacewalk.",
                "Communication systems check complete.",
                "Please send the updated flight plan.",
                "The experiment results look promising.",
                "We need additional supplies on the next resupply mission."
            ]
                
        return data

    @classmethod
    def load_test_data(cls, fname):
        # Load test data from file
        data = []
        try:
            with open(fname) as f:
                for line in f:
                    inp = line.strip()  # Remove trailing newline
                    data.append(inp)
        except Exception as e:
            print(f"Error loading test data: {e}")
            data = ["Hello"]  # Fallback
            
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        """Train the model using RoBERTa fine-tuning"""
        print(f"Training on {len(data)} sentences...")
        
        # Create work directory if it doesn't exist
        os.makedirs(work_dir, exist_ok=True)
        
        # Create an output file with sample data for training
        train_output_file = os.path.join(work_dir, 'train_data.txt')
        with open(train_output_file, 'w', encoding='utf-8') as f:
            for sentence in data:
                f.write(f"{sentence}\n")
                
        print(f"Saved training data to {train_output_file}")
        
        # Try to use train.py if it exists
        try:
            # Check for train.py in various locations
            train_path = 'train.py'
            if not os.path.exists(train_path):
                train_path = os.path.join(os.path.dirname(__file__), 'train.py')
            
            if os.path.exists(train_path):
                print(f"Found train.py at {train_path}, importing")
                
                # Import train.py
                spec = importlib.util.spec_from_file_location("train_module", train_path)
                train_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_module)
                
                # Call train function if it exists
                if hasattr(train_module, 'train'):
                    print("Using train.py for RoBERTa fine-tuning")
                    train_module.train(dataset=data)
                    print("Fine-tuning completed")
                else:
                    print("train.py doesn't have 'train' function")
            else:
                print("train.py not found")
                
                # Try to import train from installed packages
                try:
                    from train import train
                    print("Found train.py in installed packages")
                    train(dataset=data)
                    print("Fine-tuning completed")
                except ImportError:
                    print("Could not import train.py, skipping fine-tuning")
                except Exception as e:
                    print(f"Error during training: {e}")
        except Exception as e:
            print(f"Error importing train.py: {e}")
            print("Skipping fine-tuning, will use pre-trained model")
        
        print("Training process completed")

    def run_pred(self, data):
        """Run prediction on test data"""
        print(f"Predicting for {len(data)} inputs...")
        
        # Call the predict function from predict.py
        output = predict(data)
        
        return output

    def save(self, work_dir):
        # Just create a marker file - the actual model is saved by train.py
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('roberta_character_predictor')
        print(f"Model reference saved to {work_dir}")

    @classmethod
    def load(cls, work_dir):
        # Simple check if model reference exists
        model_path = os.path.join(work_dir, 'model.checkpoint')
        if os.path.exists(model_path):
            with open(model_path) as f:
                model_info = f.read()
            print(f"Model reference found: {model_info}")
        else:
            print(f"No model reference found at {model_path}")
            
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(447)
    
    # Create work directory if it doesn't exist
    os.makedirs(args.work_dir, exist_ok=True)

    if args.mode == 'train':
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        if pred:  # Check if predictions were returned
            MyModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))