import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")

parser.add_argument('--multilabel-test', 
					action='store_true', 
					help="test using a multilabel eval")

parser.add_argument('--plot', 
					action='store_true', 
					help="force plotting")

parser.add_argument('--parallel', 
					action='store_true', 
					help="parallel model")

parser.add_argument('--device', 
					type=str,
					default='cuda',
					help="the device to train the models")
parser.add_argument('--n-train', 
					type=int,
					default=5,
					help="number of trainament runs (each runs for 5 epochs)")
args = parser.parse_args()
