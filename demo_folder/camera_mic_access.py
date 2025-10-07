import cv2
import sounddevice as sd
import numpy as np
import threading

# --- CAMERA ACCESS ---
def open_camera(stop_event):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit the camera window.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Camera Feed", frame)

        # If user presses 'q', stop both camera and mic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# --- MICROPHONE ACCESS ---
def record_audio(stop_event, sample_rate=44100):
    print("Recording audio... (will stop when you press 'q')")
    recorded = []
    block_duration = 0.5  # record in small chunks

    while not stop_event.is_set():
        block = sd.rec(int(block_duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float64')
        sd.wait()
        recorded.append(block)

    print("Audio recording stopped.")
    audio = np.concatenate(recorded, axis=0)
    return audio

if __name__ == "__main__":
    stop_event = threading.Event()
    audio_result = {}

    # Start mic recording in a separate thread
    def mic_thread_func():
        audio_result["data"] = record_audio(stop_event)

    mic_thread = threading.Thread(target=mic_thread_func)
    mic_thread.start()

    # Run camera in main thread
    open_camera(stop_event)

    # Wait for mic thread to finish
    mic_thread.join()

    audio_data = audio_result.get("data", np.array([]))
    print("Audio data shape:", audio_data.shape)

