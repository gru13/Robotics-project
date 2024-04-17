# import gradio as gr

import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np


# Load the saved model
model = tf.keras.models.load_model('model/model.h5')
model = keras.models.load_model('model/model.h5')
# model = gr.load('maxineattobrah/RealTimeEmotionDetection', src='models')

# Define the face cascade and emotions
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

no_face_detection_alert = "Cannot Detect Face"
low_confidence_alert = "Cannot Detect Emotion"

# Define the predict_emotion function

def predict_emotion(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		face = gray[y:y+h, x:x+w]
		face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_AREA)
		if np.sum([face])!=0:
			face = face.astype('float')/255.0
			face = tf.keras.utils.img_to_array(face)
			face = np.expand_dims(face, axis=0)
			prediction = model.predict(face)
			if any(prob >.5 for prob in prediction[0]):
				emotion = emotions[np.argmax(prediction)]
				cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
			else:
				cv2.putText(frame, low_confidence_alert, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
	                                2)
		else:
			cv2.putText(frame, no_face_detection_alert, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	return frame

# Start the video capture and emotion detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        frame = predict_emotion(frame)
        cv2.imshow('Live Facial Emotion Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# input_image = gr.Image(source = "webcam", streaming = True, label="Your Face")
# # video = gr.inputs.Video(source = "webcam" )

# output_image = gr.Image( type="numpy", label="Detected Emotion" )



# iface = gr.Interface(
# 	fn = predict_emotion, 
# 	inputs=input_image, 
# 	outputs=output_image,
# 	batch = True,
# 	max_batch_size = 100000,
# 	# interpretation = "default",
# 	title = "Mood Detectives",
# 	description = "Real-Time Emotion Detection Using Facial Expressions:\nCan our model detect if you are angry, happy, sad, fearful, disgusted, surprised or neutral?",
# 	live = True

# 	)

# iface.queue(concurrency_count=1000)

# iface.launch()