from flask import Flask, Response
from flask_cors import CORS
import cv2
from fer import FER
from PIL import Image


app = Flask(__name__)

CORS(app)  # Enable CORS for the Flask app

def generate_frames(emotion_detector):

    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect faces
        result = emotion_detector.detect_emotions(frame)
        num_faces = len(result)
        cv2.putText(frame, f"Number of people: {num_faces}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if num_faces == 0:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            for face in result:
                (x, y, w, h) = face['box']
                emotion_text = f"Emotion: {max(face['emotions'], key=face['emotions'].get)}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert the frame to RGB format for Pillow
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Prepare the frame for display

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(emotion_detector), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    emotion_detector = FER()
    app.run(debug=False)
