<h1>Face Recognition Model Documentation</h1>

<h2>Introduction</h2>
<p>The Face Recognition model is designed to classify images into two classes: "Central Cee" and "Others." It utilizes a convolutional neural network (CNN) implemented using TensorFlow and Keras.</p>

<h2>Dependencies</h2>
<p>The following dependencies are required to run the code:</p>
<ul>
  <li><code>TensorFlow</code></li>
  <li><code>NumPy</code></li>
</ul>
<p>Ensure that these dependencies are installed before running the code. You can install them using the following command:</p>
<code>pip install tensorflow numpy</code>

<h2>Dataset</h2>
<p>The model is trained on a dataset located at the specified <code>train_data_dir</code> path. Make sure to organize your dataset into two folders - one for each class ("Central Cee" and "Others").</p>

<h2>Model Architecture</h2>
<p>The model architecture is as follows:</p>
<ul>
  <li>Input layer: Convolutional layer with 32 filters, a kernel size of (3, 3), and ReLU activation.</li>
  <li>MaxPooling layer with a pool size of (2, 2).</li>
  <li>Convolutional layer with 64 filters, a kernel size of (3, 3), and ReLU activation.</li>
  <li>MaxPooling layer with a pool size of (2, 2).</li>
  <li>Flatten layer to convert the 2D output to a 1D tensor.</li>
  <li>Dense layer with 64 units and ReLU activation.</li>
  <li>Output layer with 1 unit and Sigmoid activation.</li>
</ul>
<p>Is compiled with the Adam optimizer and binary cross-entropy loss function for binary classification. The accuracy metric is used for evaluation.</p>

<h2>Training</h2>
<p>The model is trained using an ImageDataGenerator to perform real-time data augmentation. The training data is loaded from the specified <code>train_data_dir</code>, resized to <code>img_size</code>, and normalized.</p>
<p>The training process involves fitting the model to the training data using the <code>fit</code> method with a specified number of epochs.</p>

<h2>Prediction</h2>
<p>To make predictions, an image is loaded using the <code>image.load_img</code> and converted into a format suitable for the model using <code>image.img_to_array</code>. The pixel values are normalized before making predictions.</p>
<p>The model predicts whether the input image belongs to the "Central Cee" class or "Others" based on a threshold of 0.5. The result is printed to the console.</p>

<h2>Usage</h2>
  <ol>
    <li>Organize your dataset into two folders: "Central Cee" and "Others."</li>
    <li>Set the <code>train_data_dir</code> variable to the path of your training dataset.</li>
    <li>Adjust other parameters such as <code>img_size</code> and <code>batch_size</code> as needed.</li>
    <li>Run the script to train the model.</li>
    <li>After training, use the model to make predictions on new images by specifying the image path in the <code>img_path</code> variable.</li>
  </ol>

<h2>This schema must be taken into consideration when allocating your database</h2>
    <pre>
        <code>
# - dataset/
#   - train/
#     - rapper/
#       - rapper_face_1.jpg
#       - rapper_face_2.jpg
        - ...
#     - other/
#       - other_face_1.jpg
#       - other_face_2.jpg
        - ...
        </code>
    </pre>



