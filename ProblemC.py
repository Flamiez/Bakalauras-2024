import os
import time
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
import pyaudio
import librosa

from tensorflow.signal import stft, hann_window, linear_to_mel_weight_matrix
from tensorflow.math import log, abs
from tensorflow import complex, zeros_like, matmul
from tensorflow.keras.models import load_model

from pygame import mixer
from customtkinter import *
from tqdm import tqdm
from threading import *
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.io import wavfile



directory = os.path.join(os.getcwd(), "results")
file_path = os.path.join(directory, "results.txt")
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(file_path):
    with open(file_path, 'w'):
        pass

def tensorflow_dataset_to_numpy_array(tf_dataset):
    arr = []
    for data in tf_dataset:
        arr.append(np.array(data))
    arr = np.array(arr).squeeze(axis=1)
    return arr

def scale_to_range(data, range_min=0, range_max=1):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) * (range_max - range_min) / (max_vals - min_vals) + range_min
    return normalized_data

def norm(data, type="tf"):
    try:
        ds = tensorflow_dataset_to_numpy_array(data)
    except:
        ds = data
    shape = ds.shape
    flat = ds.reshape(-1, 128)
    flat = scale_to_range(flat)
    if type=="tf":
        return tf.data.Dataset.from_tensor_slices(flat.reshape(shape[0],shape[1],shape[2])).batch(1)
    else:
        return flat.reshape(shape[0],shape[1],shape[2])

def load_sound_file(path):
    return librosa.load(path, sr=None)

def calculate_MSE_array(model, dataset):
    try:
        dataset = tensorflow_dataset_to_numpy_array(dataset)
    except:
        pass
    shape = dataset.shape
    predicted_data = model.predict(dataset)



    original_data_flat = dataset.reshape(-1, 128)
    predicted_data_flat = predicted_data.reshape(-1, 128)
    # print(original_data_flat.shape)
    # print(predicted_data_flat.shape)
    mse_per_slice = np.mean((original_data_flat - predicted_data_flat) ** 2, axis=1)
    mse_per_slice = mse_per_slice.reshape(shape[0], shape[1])

    return mse_per_slice

def create_dataset(files):
    dataset = []
    for index in tqdm(range(len(files))):
        signal, sr = load_sound_file(files[index])
        
        preprocessed = preprocess_audio(
            signal, 
            sr
            )
        dataset.append(preprocessed)

    return np.array(dataset)

def preprocess_audio(signals, sample_rate=16000, fft_length=1024):
    spectrogram = stft(
                                signals, 
                                frame_length=int(sample_rate * (0.064)), 
                                frame_step=int(sample_rate * (0.032)), 
                                fft_length=fft_length,
                                window_fn=hann_window
                                )

    mel_spectrogram = linear_to_mel_weight_matrix(num_mel_bins=128,num_spectrogram_bins=fft_length // 2 + 1)
    mel_spectrogram = complex(mel_spectrogram, zeros_like(mel_spectrogram))
    mel_spectrogram = matmul(spectrogram, mel_spectrogram)
    mel_spectrogram = log(abs(mel_spectrogram) + 1e-10)
    
    return mel_spectrogram

def predict_audio(model, path, threshold):
    classification = "None"

    data = norm(create_dataset(path))
    errors = calculate_MSE_array(model,data)
    error = np.mean(errors, axis=1)[0]
    
    if error >= threshold:
        classification = "Anomalous audio"
    else:
        classification = "Normal audio"
    return classification, error, threshold, errors[0]

def plot_and_save_image_waveform(file_path, tab_name):

    sample_rate, audio_data = wavfile.read(file_path)
    plt.figure(figsize=(6, 2))
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    image_file_path = os.path.join(os.getcwd(),f"waveform_{tab_name}.png")

    plt.savefig(image_file_path, format='png')
    return image_file_path

def plot_and_save_image_spectrogram(spectrogram, tab_name):
    spectrogram_np = np.array(spectrogram).T
    plt.figure(figsize=(6, 2))
    plt.imshow(spectrogram_np, aspect='auto', cmap='viridis', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.title('Spectrogram')
    plt.colorbar(label='Magnitude (dB)')

    image_file_path = f"spectrogram_{tab_name}.png"
    plt.savefig(image_file_path, format='png')
    return image_file_path

def plot_and_save_image_MSE(mse_list, tab_name):

    plt.figure(figsize=(6, 2))
    plt.plot(mse_list)
    plt.title('MSE error graph')
    plt.xlabel('Frames')
    plt.ylabel('Error')
    plt.tight_layout()

    image_file_path = os.path.join(os.getcwd(),f"MSE_{tab_name}.png")
    plt.savefig(image_file_path, format='png')
    return image_file_path

class AudioUI(CTk):
    def __init__(self):
        super().__init__()
        self.title("ProblemC")
        self.geometry("1600x900")
        self.model = load_model(os.path.join(os.getcwd(),'model.h5'))
        self.threshold = 0.02
        self.data_location=os.path.join(os.getcwd(),"results","results.txt")

        self.logger = Logger(self)
        self.logger.grid(row=1, column=0, sticky="nw", padx=10, pady=10) 
        
        self.microphone_frame = MicrophoneFrame(self)
        self.microphone_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self.audio_tabview = AudioTabView(self, width=650, height=600, anchor = "nw")
        self.audio_tabview.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.clear_data_button = CTkButton(self, text="Clear prediction data", command=lambda: self.clear_data())
        self.clear_data_button.grid(row=1, column=1, padx=10, pady=10, sticky="nw")

        self.clear_recordings_button = CTkButton(self, text="Delete saved recordings", command=lambda: self.clear_recordings())
        self.clear_recordings_button.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        self.input_threshold = CTkEntry(self, placeholder_text="Threshold")
        self.input_threshold.grid(row=1, column=1, padx=10, pady=10, sticky="ne")

        self.set_threshold_button = CTkButton(self, text="Set threshold", command=lambda: self.set_threshold())
        self.set_threshold_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")



    def clear_data(self):
        try:
            os.remove(self.data_location)
            with open(file_path, 'w'):
                pass
            self.logger.log("Prediction data cleared successfully")
        except:
            self.logger.log("Failed to clear prediciton data")

    def clear_recordings(self):
        try:
            if not os.path.exists(self.microphone_frame.scrollable_frame.recordings_folder):
                os.makedirs(self.microphone_frame.scrollable_frame.recordings_folder)
            shutil.rmtree(self.microphone_frame.scrollable_frame.recordings_folder, ignore_errors=True)
            if not os.path.exists(self.microphone_frame.scrollable_frame.recordings_folder):
                os.makedirs(self.microphone_frame.scrollable_frame.recordings_folder)
            if len(os.listdir(self.microphone_frame.scrollable_frame.recordings_folder)) > 0:
                self.logger.log("Some recordings failed to delete")
            else:
                self.logger.log("Recordings deleted successfully")
        except:
            self.logger.log("Failed to delete recordings")    

    def set_threshold(self):
        new = self.input_threshold.get()
        prev = self.threshold
        try:
            new_val = float(new)
            self.threshold = new_val
            self.logger.log(f"Previous threshold: {prev}, new threshold: {new}")
        except:
            self.logger.log(f"Threshold value must be a number! (use . instead of , as delimiter)")



class Logger(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, width=450, height=200)

        self.scrollable_frame = CTkScrollableFrame(self, width=450, height=200)
        self.scrollable_frame.grid(row=0, column=0, padx=10, pady=5)

        self.logger = CTkTextbox(self.scrollable_frame, wrap='word', **kwargs)
        self.logger.pack(fill='both', expand=True)

    def log(self, message):
        self.logger.configure(state='normal')
        self.logger.insert('end', message + '\n')
        self.logger.configure(state='disabled')

class AudioTabView(CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.pred = None
        self.error = None
        self.errors = None
        self.thresh = None
        self.grid(row=3, column=0)
        self.grid_propagate(False)

    def create_audio_tab(self, file_path, tab_name):
        tab = self.master.audio_tabview.add(tab_name)

        # image_buffer = io.BytesIO()
        # plt.savefig(image_buffer, format='png')
        # image_buffer.seek(0)
        # waveform_image = Image.open(image_buffer)

        image_file_path_waveform = plot_and_save_image_waveform(file_path=file_path,tab_name=tab)
        

        if os.path.exists(image_file_path_waveform):
            waveform_image = Image.open(image_file_path_waveform)
            waveform_image_widget = CTkImage(light_image=waveform_image, dark_image=waveform_image, size=(600, 200))
            CTkLabel(tab, image=waveform_image_widget, text="").grid(row=0, column=0, padx=10, pady=10)
            os.remove(image_file_path_waveform)
        else:
            self.master.logger.log("Failed to load waveform image")

        play_button = CTkButton(tab, text="Play Audio", command=lambda: self.play_audio(file_path))
        play_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        close_button = CTkButton(tab, text="Close Tab", command=lambda: self.close_tab(tab_name))
        close_button.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        predict_button = CTkButton(tab, text="Save data", command=lambda: self.save_data(tab_name))
        predict_button.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        pred_label = CTkLabel(tab, text=f"{self.pred_audio(file_path)}")
        pred_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # print(self.errors)
        # print(self.error)
        # print(self.pred)
        # print(self.thresh)
        image_file_path_MSE = plot_and_save_image_MSE(mse_list=self.errors,tab_name=tab)

        if os.path.exists(image_file_path_MSE):
            MSE_image = Image.open(image_file_path_MSE)
            MSE_image_widget = CTkImage(light_image=MSE_image, dark_image=MSE_image, size=(600, 200))
            CTkLabel(tab, image=MSE_image_widget, text="").grid(row=3, column=0, padx=10, pady=10)
            os.remove(image_file_path_MSE)
        else:
            self.master.logger.log("Failed to load errors image")

    def save_data(self, tab_name):
        data = f"{tab_name} Prediction: {self.pred} Error: {self.error} Threshold: {self.thresh}"
        self.master.logger.log(f"Saving data to results file:  {data}")
        print(self.master.data_location)
        with open(self.master.data_location, 'a') as results:
            results.write(data+"\n")

        
    def close_tab(self,tab_name):
        try:
            mixer.music.stop()
        except:
            pass
        self.master.audio_tabview.delete(tab_name)

    def play_audio(self, file_path):
        mixer.init()
        mixer.music.load(file_path)
        mixer.music.play(loops=0)

    def pred_audio(self, file_path):
        self.pred, self.error, self.thresh, self.errors = predict_audio(self.master.model, [file_path], self.master.threshold)
        stringified = f"Prediction {self.pred}\n Error {self.error}\n Threshold {self.thresh}"
        self.master.logger.log(stringified)
        return stringified

class MicrophoneFrame(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, width=400, height=500)
        self.master = master
        self.grid(row=3, column=0)
        self.microphone_label = CTkLabel(self, text="Available microphones:")
        self.microphone_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.scrollable_frame = ScrollableFrameCheckbox(self, width=400, height=500)
        self.scrollable_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

        self.update_microphones_button = CTkButton(self, text="Update microphones", command=lambda: Thread(target=self.scrollable_frame.update_available_mics).start())
        self.update_microphones_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.record_button = CTkButton(self, text="Start recording", command=lambda: Thread(target=self.scrollable_frame.start_recording).start())
        self.record_button.grid(row=2, column=2, padx=10, pady=5, sticky="e")

        self.upload_button = CTkButton(self, text="Upload audio", command=lambda: Thread(target=self.upload_audio).start())
        self.upload_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")


    def upload_audio(self):
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.wav")])
        if file_path:
            file_name = os.path.basename(file_path)
            tab_name = file_name.split("/")[-1]
            self.master.logger.log(f"Creating audio tab for {tab_name}")
            self.master.audio_tabview.create_audio_tab(file_path, tab_name)

class ScrollableFrameCheckbox(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.checkboxes = []
        self.master = master
        self.available_microphones = self.get_available_microphones()
        self.recordings_folder = os.path.join(os.getcwd(),"recordings")

        for mic in self.available_microphones:
            checkbox = CTkCheckBox(self, text=mic)
            checkbox.grid(row=len(self.checkboxes), column=0, sticky="w")
            self.checkboxes.append(checkbox)

    def update_available_mics(self):
        self.available_microphones = self.get_available_microphones()

        for checkbox in self.checkboxes:
            checkbox.destroy()
        self.checkboxes = []
        for mic in self.available_microphones:
            checkbox = CTkCheckBox(self, text=mic)
            checkbox.grid(row=len(self.checkboxes), column=0, sticky="w")
            self.checkboxes.append(checkbox)

    def get_available_microphones(self):
        p = pyaudio.PyAudio()
        available_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] != 0 and device_info['hostApi'] == 0:
                self.master.master.logger.log(f"Microphone: {device_info['name']}, Channels: {device_info['maxInputChannels']}")
                available_devices.append(device_info['name'])
        p.terminate()
        return available_devices

    def get_selected_microphones(self):
        selected_microphones = []
        for checkbox in self.checkboxes:
            if checkbox.get():
                selected_microphone = checkbox.cget("text")
                selected_microphones.append(self.available_microphones.index(selected_microphone))
        return selected_microphones
    

    def record_audio(self, mic, file_path, results):
        frames_per_buffer = 1024
        chunk = 44100 // 12
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100

        p = pyaudio.PyAudio()
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=frames_per_buffer,
                        input_device_index=mic)

        print("Recording...")
        frames = []
        for i in range(12 * rate // chunk):
            data = stream.read(chunk)
            frames.append(data)

        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

        results[mic] = file_path

    def start_recording(self):
        if not os.path.exists(self.recordings_folder):
            os.makedirs(self.recordings_folder)

        mics = self.get_selected_microphones()
        threads = {}
        results = {}

        for mic in mics:
            self.master.master.logger.log(f"Started audio recording for {self.available_microphones[mic]}")
            t = time.localtime(time.time())
            f_name = f"{self.available_microphones[mic]} {t.tm_hour}h {t.tm_min}m {t.tm_sec}s"
            file_name = f"{f_name}.wav"
            file_path = os.path.join(self.recordings_folder, file_name)

            time.sleep(0.2)

            thread = Thread(target=self.record_audio, args=(mic, file_path, results))
            thread.start()
            threads[mic] = thread

        for mic, thread in threads.items():
            self.master.master.logger.log("Waiting for all threads to complete")
            thread.join()

        self.master.master.logger.log("All recording threads finished")

        for mic, file_path in results.items():
            file_name = f"{self.available_microphones[mic]} {time.localtime(time.time()).tm_hour}h {time.localtime(time.time()).tm_min}m {time.localtime(time.time()).tm_sec}s.wav"
            self.master.master.logger.log(f"Creating audio tab for {file_name}")
            self.master.master.audio_tabview.create_audio_tab(file_path, file_name)




app = AudioUI()
app.mainloop()
