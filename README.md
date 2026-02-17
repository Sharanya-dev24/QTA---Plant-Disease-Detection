# QTA---Plant-Disease-Detection
Applying a customized CNN model to study plant image data, and classify healthy plants from diseased plants

**CONTEXT:**
#Why ReLU
'''Keeps positive values as they are
Converts negative values to zero
Why ReLU is used:
Introduces non-linearity (helps the model learn complex patterns)
Faster to compute
Reduces vanishing gradient problem'''

'''
Conv2D → extracts features like edges, textures
MaxPooling → reduces size, keeps important info
Deeper layers → learn complex patterns (disease spots)
Flatten → prepares data for classification
Dense + ReLU → decision making
Dropout (0.5) → prevents overfitting
Softmax output → predicts disease class'''

'''Block 1	32	Edges, corners, color gradients
Block 2	64	Simple shapes, leaf veins, outlines
Block 3	128	Disease spots, textures, patterns
Block 4	256	Complex disease regions & combinations'''

''' Why pooling layer after each convolution layer: 
Reduces image size
Keeps important features
Makes model faster and more robust'''

'''Why BatchNormalization: it stabilizes the learning and improves the accuracy'''

**DRAWBACKS:**
The model trains very slow - approximately an hour for each epoch. Try using Googlecolab, with the T4 Tesla, for better runs. Else reduce the complexity of images(reduce dimensions, or feed smaller dataset to the model, in larger number of batches.
