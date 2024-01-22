from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

import os
import sys
import wave
import time
import librosa
import numpy as np 
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
import speech_recognition as sr


loaded_model = tf.keras.models.load_model('./Binary_Recognizer.h5')

if os.path.exists("./temp.wav"):
    os.remove("./temp.wav")
    
class AudioRecorder(QThread):
    def __init__(self, filename, duration, sample_rate=16000):
        super().__init__()
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate

    def run(self):
        print("Recording...")
        audio_data = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()

        print("Finished recording.")

        sf.write(self.filename, audio_data, self.sample_rate)

class TextRecorder(QThread):
    recognition_complete = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        
    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=5)
        
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio, language="tr") 
            print("Text: " + text)
            self.recognition_complete.emit(text)
        
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            self.recognition_complete.emit("None")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            self.recognition_complete.emit("None")

class BasicWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        window_width = 700
        window_height = 600

        self.setGeometry(
            (screen_geometry.width() - window_width) // 2,
            (screen_geometry.height() - window_height) // 2,
            window_width,
            window_height
        )
        self.setWindowTitle('Customized PyQt5 Window')  
        self.setStyleSheet("background-color: #252729;")

        self.picture_label = QLabel(self)
        self.picture_pixmap = QPixmap("User.png")

        self.picture_pixmap = self.picture_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.picture_label.setPixmap(self.picture_pixmap)
        self.picture_label.setAlignment(Qt.AlignCenter)
        self.picture_label.setStyleSheet("border-radius: 50px; border: 2px solid #00BFFF;")

        self.button = QPushButton('Giriş Yapmak için Konuşun', self)
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px;")
        self.button.clicked.connect(self.on_button_click)
        
        # Ilk aşama. Kullanıcı tanımlama için ses alınır->
        self.recording_label = QLabel('Kayıt alınıyor...', self)
        self.recording_label.move(0, 50) 
        self.recording_label.setStyleSheet("color: black; font-size: 32px;")
        self.recording_label.setVisible(False)

        # Kullanıcı girişi sonrası->
        self.Tansel_label = QLabel('Hoşgeldiniz Tansel Akgül', self)
        self.Tansel_label.move(300, 500)  
        self.Tansel_label.setStyleSheet("color: black; font-size: 32px;")
        self.Tansel_label.setVisible(False)
        
        # Bilinmeyen Kullanıcı
        self.other_label = QLabel('Kullanıcı Bulunamadı', self)
        self.other_label.move(300, 500)  
        self.other_label.setStyleSheet("color: black; font-size: 32px;")
        self.other_label.setVisible(False)
        
        #Quit butonu
        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setStyleSheet("background-color: #4AAFF0; color: black; padding: 10px 20px; border: none; border-radius: 5px;")

        self.quit_button.clicked.connect(QApplication.instance().quit)
        #self.other_label.setVisible(False)
        self.quit_button.move(580, 550)
        
        # Buton Layout 
        layout = QVBoxLayout(self)
        layout.addWidget(self.picture_label, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout.addWidget(self.recording_label, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout.addWidget(self.Tansel_label, alignment=Qt.AlignTop | Qt.AlignHCenter)  
        layout.addWidget(self.other_label, alignment=Qt.AlignTop | Qt.AlignHCenter)  
        
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout.addLayout(button_layout)

        # Create a timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_timeout)

        self.show()

    def on_button_click(self):
        
        picture_pixmap = QPixmap("User.png")
        picture_pixmap = picture_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.picture_label.setPixmap(picture_pixmap)

        #print("Button clicked!")
        self.recording_label.setVisible(True)
        
        self.other_label.setVisible(False)
        
        # Start recording audio in a separate thread
        self.audio_recorder = AudioRecorder('temp.wav', duration=3)
        self.audio_recorder.start()
        self.Wav2Vec = TextRecorder()
        self.Wav2Vec.recognition_complete.connect(self.handle_recognition_complete)
        # Start the timer for a 3-second delay to hide the label
        self.timer.start(3200)  # 3000 milliseconds = 3 seconds
        
    def handle_recognition_complete(self, text):
        if text == "None":
            self.recording_label.setText("Komut Anlaşılmadı")
            
        else:
            self.recording_label.setText(text)
        
    def on_timer_timeout(self):
        self.timer.stop()
        self.recording_label.setVisible(False)
        features = []
    
            ##--> PREDICTION PART BEGIN
        audio, sr = librosa.load("./temp.wav", sr=16000, duration=1)
        os.remove('./temp.wav')
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        #mfccs = StandardScaler().fit_transform(mfccs)
        features.append(mfccs.T)
        
        y_pred_probabilities = loaded_model.predict(np.array(features))
        y_pred = np.argmax(y_pred_probabilities, axis=1)
                    
        if y_pred_probabilities[0][1] > 0.80:
        #if y_pred[0] == 1:
            picture_pixmap = QPixmap("Tansel.jpg")
            picture_pixmap = picture_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.picture_label.setPixmap(picture_pixmap)
            self.Tansel_label.setVisible(True)
            
            self.button.deleteLater()
            self.Komut = QPushButton('Komut Vermek için Konuşun', self)
            self.Komut.setStyleSheet("background-color: #4CAF50; color: white; padding: 100px 200px; border: none; border-radius: 5px;")
            self.Komut.clicked.connect(self.TakeAudio)
            self.Komut.setVisible(True)

            self.Komut.setFixedSize(200, 40)
            self.Komut.move(250, 400)
            
            self.other_label.setVisible(False)
            
        else:
           
            self.other_label.setVisible(True)
            
            ##--> PREDICTION PART END
    
    
    def TakeAudio(self):
        
        
        self.recording_label.setText("Komut alınıyor..")
        self.recording_label.setVisible(True)        
        self.Tansel_label.setVisible(False)
        
        text = self.Wav2Vec.start()
        
        if text:
            self.recording_label.setText(text)
            
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BasicWindow()
    sys.exit(app.exec_())
