import time
import threading
from streamlit import st
from stqdm import stqdm

# Define the task's total duration in seconds
task_duration = 10

def fake_progress_bar(duration):
    # Display a progress bar with stqdm
    for _ in stqdm(range(100), desc="Processing..."):
        time.sleep(duration / 100)  # Update bar at regular intervals

def actual_task(duration):
    # Simulate an actual task
    time.sleep(duration)  # Perform the actual task

# Initialize and start threads
progress_thread = threading.Thread(target=fake_progress_bar, args=(task_duration,))
task_thread = threading.Thread(target=actual_task, args=(task_duration,))

progress_thread.start()
task_thread.start()

# Wait for both threads to complete
progress_thread.join()
task_thread.join()

st.write("Task completed!")
