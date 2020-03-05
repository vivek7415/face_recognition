# Overview
### Detect and recognize user's face

# Steps to follow:

 1. Create a folder named 'training_image', which contains two subfolder namely '0' and '1', containg 'false' and 'true' images.
 2. Start with 'face_recognition' file which contains the face detection and face recognition logic.
 3. Move on to 'tester' file which trains the model and save the weights in 'traing_data.yml' file.
4. "tester' file also contains the logic to take live feed from the camera and allows the user to see the output.
 5. Comment the model training code once model has been trained and use the pretrained weight.
 6. 'video2img' file contains the code which helps the user to capture true images of user for training purpose.
