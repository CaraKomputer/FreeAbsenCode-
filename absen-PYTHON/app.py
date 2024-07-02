import os
import cv2
import face_recognition
from flask import Flask, render_template, Response, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Inisialisasi video capture sebagai global
video_capture = None

def initialize_video_capture():
    global video_capture
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Video capture initialized with DirectShow")

def capture_face_encodings(name):
    global video_capture
    encodings = []

    if video_capture is None:
        initialize_video_capture()

    if not video_capture.isOpened():
        flash('Kamera tidak terbuka!', 'error')
        print("Failed to open video capture")
        return False

    while len(encodings) < 20:
        ret, frame = video_capture.read()
        if not ret:
            flash('Gagal mengambil gambar saat registrasi!', 'error')
            print("Failed to read frame from video capture")
            return False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected {len(face_locations)} face(s)")

        for face_encoding in face_encodings:
            encodings.append(face_encoding)
            if len(encodings) >= 20:
                break

    if not os.path.exists('encodings'):
        os.makedirs('encodings')
    with open(f'encodings/{name}.txt', 'w') as f:
        for encoding in encodings:
            f.write(','.join([str(val) for val in encoding]) + '\n')

    print(f"Captured {len(encodings)} encodings for {name}")
    return True

def load_face_encodings():
    encodings = {}
    if not os.path.exists('encodings'):
        return encodings
    for file_name in os.listdir('encodings'):
        name = os.path.splitext(file_name)[0]
        with open(f'encodings/{file_name}', 'r') as f:
            lines = f.readlines()
            encodings[name] = [list(map(float, line.strip().split(','))) for line in lines]
    return encodings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if not name:
            flash('Nama tidak boleh kosong!', 'error')
            return redirect(url_for('register'))
        if capture_face_encodings(name):
            flash('Pendaftaran berhasil!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    global video_capture
    if request.method == 'POST':
        known_encodings = load_face_encodings()
        if video_capture is None:
            initialize_video_capture()

        if not video_capture.isOpened():
            flash('Kamera tidak terbuka!', 'error')
            print("Failed to open video capture for recognition")
            return redirect(url_for('recognize'))

        ret, frame = video_capture.read()
        if not ret:
            flash('Gagal mengambil gambar!', 'error')
            print("Failed to read frame from video capture for recognition")
            return redirect(url_for('recognize'))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected {len(face_locations)} face(s) during recognition")

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                [encoding for name, encodings in known_encodings.items() for encoding in encodings], 
                face_encoding
            )
            name = "Tidak Dikenal"
            if True in matches:
                match_index = matches.index(True)
                name = list(known_encodings.keys())[match_index // 20]
            flash(f'Wajah dikenali sebagai {name}', 'success' if name != "Tidak Dikenal" else 'error')
            return redirect(url_for('index'))
    return render_template('recognize.html')

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    if request.method == 'POST':
        name = request.form['name']
        if not name:
            flash('Nama tidak boleh kosong!', 'error')
            return redirect(url_for('delete'))

        # Hapus file encoding
        file_path = f'encodings/{name}.txt'
        if os.path.exists(file_path):
            os.remove(file_path)
            flash('Data wajah berhasil dihapus!', 'success')
        else:
            flash('Data wajah tidak ditemukan!', 'error')

        return redirect(url_for('index'))

    return render_template('delete.html')

def gen_frames():
    global video_capture
    if video_capture is None:
        initialize_video_capture()
    if not video_capture.isOpened():
        print("Kamera tidak terbuka!")
    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to read frame from video capture for streaming")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_camera')
def view_camera():
    return render_template('view_camera.html')

if __name__ == '__main__':
    app.run(debug=True)
