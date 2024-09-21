import numpy as np
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from utils.model import CNNModel, generate_caption_beam_search
import os

from config import config

from gtts import gTTS
import datetime

"""
    *Some simple checking
"""
assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'

# Extract features from each image in the directory
def extract_features(filename, model, model_type):
	if model_type == 'inceptionv3':
		from keras.applications.inception_v3 import preprocess_input
		target_size = (299, 299)
	elif model_type == 'vgg16':
		from keras.applications.vgg16 import preprocess_input
		target_size = (224, 224)
	# Loading and resizing image
	image = load_img(filename, target_size=target_size)
	# Convert the image pixels to a numpy array
	image = img_to_array(image)
	# Reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# Prepare the image for the CNN Model model
	image = preprocess_input(image)
	# Pass image into model to get encoded features
	features = model.predict(image, verbose=0)
	return features


# Load the tokenizer
tokenizer_path = config['tokenizer_path']
tokenizer = load(open(tokenizer_path, 'rb'))

# Max sequence length (from  training)
max_length = config['max_length']

# Load the model
caption_model = load_model(config['model_load_path'])

image_model = CNNModel(config['model_type'])


# Converting the text captions to audio format
def cap_to_audio(captions):
	lang="en"
	# Creating an instance of gTTS
	tts = gTTS(text=caption, lang=lang, slow=False)

	folder_path=config['audio_path']

	#We make sure if the folder exists, if not we make a new folder
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	# Full path for the output file
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	file_path= os.path.join(folder_path,"output_"+timestamp+"_"+".mp3")	
	# file_path=os.path.join(folder_path,"output.mp3")

	#Save the file into a specified location
	tts.save(file_path)

	print(f"Audio file saved as {file_path}")

# Load and prepare the image
for image_file in os.listdir(config['test_data_path']):
	if(image_file.split('--')[0]=='output'):
		continue
	if '.' in image_file and (image_file.split('.')[-1] == 'jpg' or image_file.split('.')[-1] == 'jpeg'):
		print('Generating caption for {}'.format(image_file))
		# Encode image using CNN Model
		image = extract_features(config['test_data_path']+image_file, image_model, config['model_type'])
		# Generate caption using Decoder RNN Model + BEAM search
		generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k'])
		# Remove startseq and endseq
		caption = 'Caption: ' + generated_caption.split()[1].capitalize()
		for x in generated_caption.split()[2:len(generated_caption.split())-1]:
			caption = caption + ' ' + x
		caption += '.'

		# print(caption)
		cap_to_audio(caption)
		# Show image and its caption
		pil_im = Image.open(config['test_data_path']+image_file, 'r')
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
		_ = ax.set_title("BEAM Search with k={}\n{}".format(config['beam_search_k'],caption),fontdict={'fontsize': '20','fontweight' : '40'})
		plt.savefig(config['test_data_path']+'output--'+image_file)

# import numpy as np
# from PIL import Image
# from pickle import load
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# from utils.model import CNNModel, generate_caption_beam_search
# import os
# from nltk.translate.bleu_score import sentence_bleu  # Import for BLEU score

# from config import config

# from gtts import gTTS
# import datetime

# # Some simple checking
# assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
# assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'

# # Extract features from each image in the directory
# def extract_features(filename, model, model_type):
#     if model_type == 'inceptionv3':
#         from keras.applications.inception_v3 import preprocess_input
#         target_size = (299, 299)
#     elif model_type == 'vgg16':
#         from keras.applications.vgg16 import preprocess_input
#         target_size = (224, 224)
#     # Loading and resizing image
#     image = load_img(filename, target_size=target_size)
#     # Convert the image pixels to a numpy array
#     image = img_to_array(image)
#     # Reshape data for the model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # Prepare the image for the CNN Model model
#     image = preprocess_input(image)
#     # Pass image into model to get encoded features
#     features = model.predict(image, verbose=0)
#     return features

# # Function to calculate BLEU score
# def calculate_bleu(reference_caption, generated_caption):
#     reference_caption = reference_caption.lower().split()  # Tokenize reference
#     generated_caption = generated_caption.lower().split()  # Tokenize generated caption
#     return sentence_bleu([reference_caption], generated_caption)

# # Load the tokenizer
# tokenizer_path = config['tokenizer_path']
# tokenizer = load(open(tokenizer_path, 'rb'))

# # Max sequence length (from training)
# max_length = config['max_length']

# # Load the model
# caption_model = load_model(config['model_load_path'])

# image_model = CNNModel(config['model_type'])

# # Converting the text captions to audio format
# def cap_to_audio(captions):
#     lang = "en"
#     # Creating an instance of gTTS
#     tts = gTTS(text=caption, lang=lang, slow=False)

#     folder_path = config['audio_path']

#     # We make sure if the folder exists, if not we make a new folder
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     # Full path for the output file
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_path = os.path.join(folder_path, "output_" + timestamp + "_" + ".mp3")

#     # Save the file into a specified location
#     tts.save(file_path)

#     print(f"Audio file saved as {file_path}")

# # Load and prepare the image
# for image_file in os.listdir(config['test_data_path']):
#     if(image_file.split('--')[0] == 'output'):
#         continue
#     if '.' in image_file and (image_file.split('.')[-1] == 'jpg' or image_file.split('.')[-1] == 'jpeg'):
#         print('Generating caption for {}'.format(image_file))
#         # Encode image using CNN Model
#         image = extract_features(config['test_data_path']+image_file, image_model, config['model_type'])
#         # Generate caption using Decoder RNN Model + BEAM search
#         generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k'])
#         # Remove startseq and endseq
#         caption = ' '.join(generated_caption.split()[1:-1])
        
#         # Assume reference captions are stored in a file or a dictionary for the test images
#         reference_caption = "your reference caption here"  # Replace with actual reference
        
#         # Calculate BLEU score
#         bleu_score = calculate_bleu(reference_caption, caption)
        
#         print(f"BLEU score for {image_file}: {bleu_score}")

#         cap_to_audio(caption)
#         # Show image, its caption, and BLEU score
#         pil_im = Image.open(config['test_data_path'] + image_file, 'r')
#         fig, ax = plt.subplots(figsize=(8, 8))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         _ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
#         _ = ax.set_title(f"BEAM Search with k={config['beam_search_k']}\n{caption}\nBLEU Score: {bleu_score:.4f}", 
#                          fontdict={'fontsize': '20', 'fontweight': '40'})
#         plt.savefig(config['test_data_path'] + 'output--' + image_file)
