# import cv2
# import sounddevice as sd
# import numpy as np
# import threading

# # --- CAMERA ACCESS ---
# def open_camera(stop_event):
#     cap = cv2.VideoCapture(0)  # 0 = default webcam
#     if not cap.isOpened():
#         print("Error: Could not access the camera.")
#         return

#     print("Press 'q' to quit the camera window.")
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame.")
#             break

#         cv2.imshow("Camera Feed", frame)

#         # If user presses 'q', stop both camera and mic
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             stop_event.set()
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # --- MICROPHONE ACCESS ---
# def record_audio(stop_event, sample_rate=44100):
#     print("Recording audio... (will stop when you press 'q')")
#     recorded = []
#     block_duration = 0.5  # record in small chunks

#     while not stop_event.is_set():
#         block = sd.rec(int(block_duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float64')
#         sd.wait()
#         recorded.append(block)

#     print("Audio recording stopped.")
#     audio = np.concatenate(recorded, axis=0)
#     return audio

# if __name__ == "__main__":
#     stop_event = threading.Event()
#     audio_result = {}

#     # Start mic recording in a separate thread
#     def mic_thread_func():
#         audio_result["data"] = record_audio(stop_event)

#     mic_thread = threading.Thread(target=mic_thread_func)
#     mic_thread.start()

#     # Run camera in main thread
#     open_camera(stop_event)

#     # Wait for mic thread to finish
#     mic_thread.join()

#     audio_data = audio_result.get("data", np.array([]))
#     print("Audio data shape:", audio_data.shape)






# camera_mic_with_db.py
import cv2
import sounddevice as sd
import numpy as np
import threading
import datetime
import mysql.connector
import soundfile as sf
import platform
import subprocess
import time

# ----------------- CONFIG ----------------- #
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "device_db",   # database created in step 2
    "port": 3306
}
AUDIO_FILENAME = "recorded_audio.wav"
CAMERA_INDEX_TO_USE = None  # If you want to force a camera index (int), set here; otherwise None = use default/probed
BLOCK_DURATION = 0.5  # seconds per audio block
SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
VIDEO_PROBE_MAX = 6
# ------------------------------------------ #

# --------- Device enumeration functions ---------
def get_audio_input_devices():
    """Return list of input audio devices and detected default input index (if available)."""
    devices = sd.query_devices()
    input_devices = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            input_devices.append({
                "index": idx,
                "name": dev.get("name", "<unknown>"),
                "max_input_channels": dev.get("max_input_channels", 0)
            })
    # get default input index (sd.default.device is (in, out) or int)
    default_in = None
    try:
        d = sd.default.device
        if isinstance(d, (list, tuple)) and len(d) >= 1:
            default_in = d[0]
        elif isinstance(d, int):
            default_in = d
    except Exception:
        default_in = None
    return input_devices, default_in

def get_video_devices(max_test=VIDEO_PROBE_MAX):
    """
    Try platform-specific ways to get camera names.
    Fallback: probe indices 0..max_test-1 and return those that open.
    Returns list of dicts {index, name}.
    """
    system = platform.system()
    devices = []

    # Windows: try pygrabber (DirectShow)
    if system == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            names = graph.get_input_devices()
            for i, name in enumerate(names):
                devices.append({"index": i, "name": name})
            if devices:
                return devices
        except Exception:
            # pygrabber not available or failed -> fallback to probing
            pass

    # Linux: try v4l2-ctl if available (v4l-utils)
    if system == "Linux":
        try:
            res = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=False)
            out = res.stdout.strip()
            if out:
                # Parse lines like:
                # "HD WebCam (usb-0000:00:14.0-4):\n\t/dev/video0\n\t/dev/video1\n                lines = out.splitlines()
                i = 0
                while i < len(lines):
                    name = lines[i].strip()
                    i += 1
                    while i < len(lines) and lines[i].startswith("\t"):
                        path = lines[i].strip()
                        devices.append({"index": path, "name": name})
                        i += 1
                if devices:
                    return devices
        except Exception:
            pass

    # Generic fallback: probe camera indices
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        # device opened
        devices.append({"index": i, "name": f"Camera index {i}"})
        cap.release()
    return devices

# -------------- MySQL helpers ---------------
def connect_db(conf):
    return mysql.connector.connect(
        host=conf["host"], user=conf["user"], password=conf["password"],
        database=conf["database"], port=conf.get("port", 3306)
    )

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_type VARCHAR(20),
            device_index VARCHAR(100),
            device_name VARCHAR(255),
            extra_info TEXT,
            detected_at DATETIME
        )
    """)
    conn.commit()

def insert_device(conn, device_type, device_index, device_name, extra_info=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO devices (device_type, device_index, device_name, extra_info, detected_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (device_type, str(device_index), device_name, extra_info, datetime.datetime.now()))
    conn.commit()
    return cur.lastrowid

# -------------- Recording functions --------------
def record_audio(stop_event, sample_rate=SAMPLE_RATE, channels=AUDIO_CHANNELS, block_duration=BLOCK_DURATION):
    print("[audio] Recording... (will stop when you press 'q' in the camera window)")
    chunks = []
    try:
        while not stop_event.is_set():
            block = sd.rec(int(block_duration * sample_rate), samplerate=sample_rate, channels=channels, dtype="float32")
            sd.wait()  # block for this chunk
            chunks.append(block)
    except Exception as e:
        print("[audio] Error while recording:", e)

    if chunks:
        audio = np.concatenate(chunks, axis=0)
    else:
        audio = np.empty((0, channels), dtype=np.float32)
    return audio, sample_rate

def open_camera_and_preview(stop_event, camera_index=None):
    idx = camera_index if camera_index is not None else 0
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[video] Could not open camera index {idx}.")
        return

    print("[video] Press 'q' in the camera window to stop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[video] Failed to read frame.")
            break
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- Main ----------------
if __name__ == "__main__":
    # 1) Enumerate devices
    print("Enumerating audio devices...")
    audio_devices, default_audio_idx = get_audio_input_devices()
    for d in audio_devices:
        print("  -", d)

    print("Enumerating video devices...")
    video_devices = get_video_devices()
    for d in video_devices:
        print("  -", d)

    # 2) Save devices to MySQL
    try:
        conn = connect_db(DB_CONFIG)
        init_db(conn)
        print("[db] Connected to MySQL, inserting detected devices...")

        for d in audio_devices:
            insert_device(conn, "audio", d["index"], d["name"], extra_info=f"max_channels={d['max_input_channels']}")
        # If default input index known, record it as an extra record
        if default_audio_idx is not None:
            insert_device(conn, "audio_default", default_audio_idx, f"default_input_index_{default_audio_idx}")

        # Video devices
        for d in video_devices:
            insert_device(conn, "video", d["index"], d["name"])
    except mysql.connector.Error as err:
        print("[db] MySQL error:", err)
        conn = None

    # 3) Start recording (camera + mic)
    stop_event = threading.Event()
    audio_result = {}

    def mic_thread_fn():
        audio_result["data"], audio_result["sr"] = record_audio(stop_event)

    mic_thread = threading.Thread(target=mic_thread_fn, daemon=True)
    mic_thread.start()

    # choose camera index to use:
    cam_index_used = CAMERA_INDEX_TO_USE
    if cam_index_used is None and video_devices:
        # if video_devices reported indices like int or '/dev/video0'
        first = video_devices[0]['index']
        # if index is a path (string starting with /dev) try to extract number, else use 0
        if isinstance(first, int):
            cam_index_used = first
        else:
            try:
                # e.g. '/dev/video0' -> 0
                if isinstance(first, str) and "video" in first:
                    cam_index_used = int(first.rsplit("video", 1)[1])
                else:
                    cam_index_used = 0
            except Exception:
                cam_index_used = 0
    else:
        cam_index_used = cam_index_used if cam_index_used is not None else 0

    # Run camera in main thread (so OpenCV window's waitKey works reliably)
    open_camera_and_preview(stop_event, camera_index=cam_index_used)

    # Wait for mic thread to finish
    mic_thread.join(timeout=2.0)

    audio_data = audio_result.get("data", np.empty((0, AUDIO_CHANNELS), dtype=np.float32))
    sr = audio_result.get("sr", SAMPLE_RATE)
    print("[main] Audio shape:", audio_data.shape)

    # Save audio to file
    try:
        if audio_data.size > 0:
            sf.write(AUDIO_FILENAME, audio_data, sr)
            print(f"[main] Saved audio to {AUDIO_FILENAME}")
            if conn:
                insert_device(conn, "recording", "", f"saved_audio:{AUDIO_FILENAME}")
    except Exception as e:
        print("[main] Could not save audio:", e)

    if conn:
        conn.close()
        print("[db] Closed DB connection")

    print("Done.")
