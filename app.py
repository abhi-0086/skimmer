from flask import Flask, request, jsonify 
import tensorflow as tf 
import numpy as np 
from preprocess import split_sentences, tokenize_sentence, tokenize_sentences, char_vectorize_sentence, char_vectorize_sentences 
app = Flask(__name__) 
# Load TensorFlow model 
model = tf.keras.models.load_model("skimlit") 
# Define label categories 
labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS'] 
# Define POST route 
@app.route('/classify_abstract', methods=['POST']) 
def classify_abstract(): 
    # Get abstract from request 
    abstract = request.json['abstract'] 
    # Split abstract into sentences 
    sentences = split_sentences(abstract) 
    total_lines = len(sentences) 
    categorized_abstract = [] 

    for idx, sentence in enumerate(sentences, start=1): 
        # Tokenize sentence 
        #tokens = tokenize_sentence(sentence) 

        # Convert tokens to character vectors 
        char_vectors = char_vectorize_sentence(sentence) 

        # Prepare inputs 
        ln_input = tf.one_hot([idx], depth=15) 
        tl_input = tf.one_hot([total_lines], depth=20) 

        # Ensure that the input variables have the correct dimensions and data types 
        token_input = tf.constant([sentence]) 
        char_input = tf.constant([char_vectors]) 
        
        # Reshape the input tensors to match the model's input signature 
        ln_input = tf.reshape(ln_input, shape=(1, 15)) 
        tl_input = tf.reshape(tl_input, shape=(1, 20)) 
        
        # Predict using the modified input data 
        prediction = model.predict([[token_input], [char_input], ln_input, tl_input]) 
        
        # Interpret predictions 
        predicted_label = labels[np.argmax(prediction)] 
        
        # Add categorized sentence to the response object 
        categorized_abstract.append(
            { "line_no": idx, 
              "sentence": sentence, 
              "category": predicted_label 
            }
        ) 
        
        # Construct response object 
        response = { "return_code": "200", "data": categorized_abstract } 
        # Send response back 
        return jsonify(response) 

if __name__ == '_main_': 
    app.run(debug=True)