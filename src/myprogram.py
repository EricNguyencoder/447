# #!/usr/bin/env python
# import os
# import string
# import random
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from predict import predict


# class MyModel:
#     """
#     This is a starter model to get you started. Feel free to modify this file.
#     """

#     @classmethod
#     def load_training_data(cls):
#         # your code here
#         # this particular model doesn't train
#         data = []
#         import csv
#         with open('generated_sentences.csv', 'r', encoding='utf-8') as f:
#             csv_reader = csv.reader(f)
#             # Skip header if your CSV has one
#             # next(csv_reader)  # Uncomment this line if your CSV has a header row
#             for row in csv_reader:
#                 # Assuming each sentence is in the first column
#                 # If your CSV structure is different, adjust the index accordingly
#                 data.append(row[0])
                
#         return data

#     @classmethod
#     def load_test_data(cls, fname):
#         # your code here
#         data = []
#         with open(fname) as f:
#             for line in f:
#                 inp = line[:-1]  # the last character is a newline
#                 data.append(inp)
#         return data

#     @classmethod
#     def write_pred(cls, preds, fname):
#         with open(fname, 'wt') as f:
#             for p in preds:
#                 f.write('{}\n'.format(p))

#     def run_train(self, data, work_dir):
#         # your code here
#         pass

#     def run_pred(self, data):
#         # your code here
#         # preds = []
#         # all_chars = string.ascii_letters
#         # for inp in data:
#         #     # this model just predicts a random character each time
#         #     top_guesses = [random.choice(all_chars) for _ in range(3)]
#         #     preds.append(''.join(top_guesses))
#         # return preds
#         predict(data)

#     def save(self, work_dir):
#         # your code here
#         # this particular model has nothing to save, but for demonstration purposes we will save a blank file
#         with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
#             f.write('dummy save')

#     @classmethod
#     def load(cls, work_dir):
#         # your code here
#         # this particular model has nothing to load, but for demonstration purposes we will load a blank file
#         with open(os.path.join(work_dir, 'model.checkpoint')) as f:
#             dummy_save = f.read()
#         return MyModel()


# if __name__ == '__main__':
#     parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument('mode', choices=('train', 'test'), help='what to run')
#     parser.add_argument('--work_dir', help='where to save', default='work')
#     parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
#     parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
#     args = parser.parse_args()

#     random.seed(447)

#     if args.mode == 'train':
#         if not os.path.isdir(args.work_dir):
#             print('Making working directory {}'.format(args.work_dir))
#             os.makedirs(args.work_dir)
#         print('Instatiating model')
#         model = MyModel()
#         print('Loading training data')
#         train_data = MyModel.load_training_data()
#         print('Training')
#         model.run_train(train_data, args.work_dir)
#         print('Saving model')
#         model.save(args.work_dir)
#     elif args.mode == 'test':
#         print('Loading model')
#         # model = MyModel.load(args.work_dir)
#         model = MyModel()
#         # print('Loading test data from {}'.format(args.test_data))
#         test_data = MyModel.load_test_data(args.test_data)
#         print('Making predictions')
#         pred = model.run_pred(test_data)
#         print('Writing predictions to {}'.format(args.test_output))
#         # assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
#         # model.write_pred(pred, args.test_output)
#     else:
#         raise NotImplementedError('Unknown mode {}'.format(args.mode))
#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from predict import predict

class MyModel:
    """
    Improved model for next character prediction.
    """

    @classmethod
    def load_training_data(cls):
        # Look for CSV in parent directory if not in current directory
        data = []
        csv_paths = [
            'generated_sentences.csv',                  # Current directory
            '../generated_sentences.csv',               # Parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated_sentences.csv')  # Absolute path to parent
        ]
        
        for csv_path in csv_paths:
            try:
                import csv
                print(f"Trying to load CSV from: {csv_path}")
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    # Skip header if your CSV has one
                    # next(csv_reader)  # Uncomment this line if your CSV has a header row
                    for row in csv_reader:
                        # Assuming each sentence is in the first column
                        if row and len(row) > 0:
                            data.append(row[0])
                    
                print(f"Successfully loaded {len(data)} sentences from {csv_path}")
                break  # Exit the loop if successful
            except Exception as e:
                print(f"Could not load from {csv_path}: {e}")
        
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
        """Train the model on provided data"""
        print(f"Training on {len(data)} sentences...")
        
        # Create an output file with sample data from training
        # This will help your predict.py function learn character patterns
        train_output_file = os.path.join(work_dir, 'train_data.txt')
        with open(train_output_file, 'w', encoding='utf-8') as f:
            for sentence in data:
                f.write(f"{sentence}\n")
                
        print(f"Saved training data to {train_output_file}")
        
        # You can add additional training logic here if needed
        
        print("Training completed")

    def run_pred(self, data):
        """Run prediction on test data"""
        print(f"Predicting for {len(data)} inputs...")
        
        # Call the predict function from predict.py
        output = predict(data)
        
        return output

    def save(self, work_dir):
        # Save model information to the work directory
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('character_predictor_v1')
        print(f"Model saved to {work_dir}")

    @classmethod
    def load(cls, work_dir):
        # Load model from work directory
        model_path = os.path.join(work_dir, 'model.checkpoint')
        if os.path.exists(model_path):
            with open(model_path) as f:
                model_info = f.read()
            print(f"Loaded model: {model_info}")
        else:
            print(f"No model found at {model_path}, initializing new model")
            
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