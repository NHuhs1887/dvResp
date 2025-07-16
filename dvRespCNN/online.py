import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from aedat_dataset import AEDATRespirationDataset
from models.dvRespNet import EventRespirationNet
import numpy as np
import dv_processing as dv
import tonic.transforms as TT
import time
from datetime import timedelta
import os
from tonic.io import read_aedat4
import threading

DATA_DIR = "../data/042025"
CSV_PATH = "../data/ground_truth.csv"
CHECKPOINT_PATH = "dvRespCNN/cpks/best_model-epoch=79-val_loss=0.42.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

capture = dv.io.CameraCapture()
filter_chain = dv.EventFilterChain()
filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(capture.getEventResolution(), backgroundActivityDuration=timedelta(milliseconds=1)))

event_counts = []
timestamps = []

latest_prediction = None
prediction_lock = threading.Lock()

writer = None
writer_lock = threading.Lock()
stop_flag = threading.Event()

recording_allowed = threading.Event()
recording_allowed.set()  # Allow recording initially


from threading import Lock
plot_lock = Lock()

def saveEvents(events: dv.EventStore):
    global writer, event_counts, timestamps
    filter_chain.accept(events)
    filtered = filter_chain.generateEvents()
    if filtered.size() > 0:
        with writer_lock:
            if writer is not None:
                writer.writeEvents(filtered)

        with plot_lock:
            event_counts.append(filtered.size())
            timestamps.append(time.time())


def transform_events(events, max_time=10, frames_per_sample=100, sensor_size=(640, 480, 2)):
    transform = TT.Compose([
        TT.CropTime(max=events['t'][0] + max_time * 1_000_000),
        TT.ToFrame(sensor_size=sensor_size, n_time_bins=frames_per_sample),
    ])
    return transform(events)


def geteventsfromfile(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    return read_aedat4(filename)


def record_events():
    global writer
    print("Starting event recording thread")
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=10), saveEvents)

    while not stop_flag.is_set():
        with writer_lock:
            if writer is None and recording_allowed.is_set():
                print("Writer is None and recording allowed, recreating writer...")
                writer = dv.io.MonoCameraWriter("temp.aedat4", capture)

        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

        time.sleep(0.001)  # avoid CPU overload


def predict_from_file():
    global writer, latest_prediction
    model = EventRespirationNet.load_from_checkpoint(CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    while not stop_flag.is_set():
        time.sleep(10)

        recording_allowed.clear()

        with writer_lock:
            print("Deleting old writer...")
            del writer
            writer = None

        print("Processing events...")
        starttime = time.time()

        processedevents = geteventsfromfile("temp.aedat4")
        if processedevents is None:
            continue

        try:
            with torch.no_grad():
                frames = torch.from_numpy(transform_events(processedevents)).float().to(DEVICE)
                frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
                y_hat = model(frames)
                print("Respiration rate prediction:", y_hat.item())
                with prediction_lock:
                    latest_prediction = y_hat.item()
        except Exception as e:
            print("Error during prediction:", e)

        try:
            os.remove("temp.aedat4")
        except FileNotFoundError:
            pass

        recording_allowed.set()

        print("Processing time:", round(time.time() - starttime, 2), "seconds")


def live_plot():
    plt.ion()
    global latest_prediction
    fig, ax = plt.subplots()
    ax.set_title("Events per Slice (Last 10 Seconds)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Event Count")
    line, = ax.plot([], [], lw=2)

    prediction_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    while not stop_flag.is_set():
        current_time = time.time()

        with plot_lock:
            # Trim old entries beyond 30 seconds
            while timestamps and current_time - timestamps[0] > 10:
                timestamps.pop(0)
                event_counts.pop(0)

            if timestamps and event_counts:
                min_len = min(len(timestamps), len(event_counts))
                trimmed_timestamps = timestamps[:min_len]
                trimmed_event_counts = event_counts[:min_len]

                x_vals = [t - trimmed_timestamps[0] for t in trimmed_timestamps]
                y_vals = trimmed_event_counts

                line.set_xdata(x_vals)
                line.set_ydata(y_vals)
                ax.relim()
                ax.autoscale_view()

        with prediction_lock:
            if latest_prediction is not None:
                prediction_text.set_text(f"Respiration rate: {latest_prediction:.2f} bpm")
            else:
                prediction_text.set_text("Waiting for prediction...")

        plt.draw()
        plt.pause(0.1)



def main():
    try:
        os.remove("temp.aedat4")
    except FileNotFoundError:
        pass

    # Start background threads for recording and prediction
    threads = [
        threading.Thread(target=record_events),
        threading.Thread(target=predict_from_file)
    ]
    for t in threads:
        t.start()

    # Run live_plot in the main thread (required for Matplotlib GUI)
    try:
        live_plot()
    except KeyboardInterrupt:
        print("Stopping...")
        stop_flag.set()

    # Wait for threads to finish
    for t in threads:
        t.join()

    with writer_lock:
        if writer:
            del writer
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
