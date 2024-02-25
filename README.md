# Sign-Language-Detection-LSTM

## Introduction
Hand gesture recognition, also known as sign language recognition, has gained increasing attention in recent years due to its potential applications in various fields such as education, entertainment, healthcare, and communication with people who have hearing and speech impairments. Deep learning algorithms have emerged as powerful tools for tackling this problem due to their ability to automatically extract relevant features from raw sensor data.

This project aims to develop a hand gesture recognition system using deep learning algorithms. We will use state-of-the-art techniques like convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to classify gestures captured via depth maps or skeleton joint positions obtained from cameras or motion capture devices. This approach will involve preprocessing the raw image data to segment hands from backgrounds, detect key points, and generate feature representations suitable for input into our deep learning models. Additionally, it will incorporate a confidence score to detect the most probable hand gesture movement.

# Simplified Methodology
1. Collect a dataset of live training data, comprising hand gestures recorded from various camera angles, to enhance the model's robustness.
2. Preprocess it by resizing the frames, normalizing the pixel intensities, and splitting the dataset into training, validation, and testing sets.
3. Create labels(Hello, I Love You, Thanks) from the preprocessed data to represent the hand gestures.
4. Build an LSTM model,where we feed them in the preprocessed data and train the model.
```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 30, 64)            442112    
                                                                 
 lstm_1 (LSTM)               (None, 30, 128)           98816     
                                                                 
 lstm_2 (LSTM)               (None, 64)                49408     
                                                                 
 dense (Dense)               (None, 64)                4160      
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 3)                 99        
                                                                 
=================================================================
Total params: 596,675
Trainable params: 596,675
Non-trainable params: 0
_________________________________________________________________
```
5. Calculate the accuracy, precision and also generate a confusion matrix to visualize the model's predictions.


## Results

<table>
  <tr>
    <td>
      <figure>
        <img src="/Results/Accuracy/Categorical Accuracy.png" alt="Accuracy" title="Accuracy">
        <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Training  Accuracy</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/Results/Accuracy/Loss.png" alt="Loss" title="Loss">
        <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Training Loss</figcaption>
      </figure>
    </td>
  </tr>
</table>

# Confusion Matrix
|              | Predicted Hello | Predicted Not Hello |
|--------------|------------------|--------------------|
| **True Hello**     | 3          | 1                  |
| **True Not Hello** | 0          | 1                  |

|              | Predicted Thanks | Predicted Not Thanks|
|--------------|------------------|--------------------|
| **True Thanks**     | 4          | 0                 |
| **True Not Thanks** | 0          | 1                 |

|              | Predicted I Love You | Predicted I Love You |
|--------------|------------------|--------------------|
| **True I Love You**     | 2          | 0                  |
| **True Not I Love You** | 1          | 2                  |

# Inference

<table>
  <tr>
    <td>
      <figure>
        <img src="/Results/Inference/Hello.png" alt="Hello" title="Hello">
        <figcaption>Hello</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/Results/Inference/I Love You.png" alt="I Love You" title="I Love You">
        <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I Love You</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/Results/Inference/Thanks.png" alt="Thanks" title="Thanks.png">
        <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Thanks</figcaption>
      </figure>
    </td>
  </tr>
</table>
