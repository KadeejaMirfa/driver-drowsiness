import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
from playsound import playsound
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import base64
import os
from streamlit.components.v1 import html

# Set page config for a cleaner look
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-container {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Driver Drowsiness Detection System")
st.markdown("Real-time monitoring system for driver safety")

# Initialize session state for alerts and sound
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
    st.session_state.last_sound_time = 0

# Create layout columns
col1, col2 = st.columns([2, 1])

with col1:
    # Video feed placeholder
    video_placeholder = st.empty()

with col2:
    # Sound component placeholder
    sound_placeholder = st.empty()
    
    # Metrics section
    st.subheader("Real-time Metrics")
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        metrics_cols = st.columns(3)
        ear_metric = metrics_cols[0].empty()
        mar_metric = metrics_cols[1].empty()
        tilt_metric = metrics_cols[2].empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Alerts section
    st.subheader("Alerts")
    alerts_placeholder = st.empty()

# Function to play alert sound using JavaScript
def play_alert_sound():
    current_time = time.time()
    
    # Prevent sound spamming - only play once every 3 seconds
    if current_time - st.session_state.last_sound_time < 3:
        return
    
    # Update the last sound time
    st.session_state.last_sound_time = current_time
    
    # Create JavaScript to play the sound
    js_code = """
    <script>
    const audio = new Audio("data:audio/wav;base64,{0}");
    audio.volume = 1.0;
    audio.play();
    </script>
    """
    
    # Read and encode the audio file
    audio_file = open('beep.wav', 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Insert the JavaScript with the base64 audio
    html(js_code.format(audio_base64), height=0)

# Initialize face detection components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Initialize video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Constants
EYE_AR_THRESH = 0.20
MOUTH_AR_THRESH = 0.79
HEAD_TILT_THRESH = 22
EYE_AR_CONSEC_FRAMES = 3
LONG_CLOSED_FRAMES = 150
HEAD_TILT_CONSEC_FRAMES = 90
ALERT_DURATION = 5  # Duration in seconds for alerts to stay visible

# Get facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize variables
COUNTER = 0
LONG_COUNTER = 0
HEAD_TILT_COUNTER = 0
CONTINUOUS_HEAD_TILT_COUNTER = 0

# Multiple events tracking
blink_times = []
yawn_times = []
head_tilt_times = []

# Flags to track if an alert has already been played
blink_alert_played = False
yawn_alert_played = False
head_tilt_alert_played = False

# Alert display timers
blink_alert_time = 0
yawn_alert_time = 0
head_tilt_alert_time = 0
eyes_closed_alert_time = 0
head_tilt_too_long_alert_time = 0

# New variables for 2-second delay in alerts
blink_condition_met_time = None
yawn_condition_met_time = None
head_tilt_condition_met_time = None
ALERT_DELAY = 2.0  # 2 seconds delay

def check_time_window(times_list, window_size=30):
    """Remove events older than the window size and return count of recent events"""
    current = time.time()
    # Keep only events within the time window
    times_list[:] = [t for t in times_list if (current - t) <= window_size]
    return len(times_list)

def reset_alert_counters(alert_type):
    """Reset specific alert counters after an alert has been triggered"""
    current_time = time.time()
    if alert_type == "blink":
        blink_times.clear()
        return current_time
    elif alert_type == "yawn":
        yawn_times.clear()
        return current_time
    elif alert_type == "head_tilt":
        head_tilt_times.clear()
        return current_time
    elif alert_type == "eyes_closed":
        return current_time
    elif alert_type == "head_tilt_long":
        return current_time
    return 0

def add_alert(alert_text):
    """Add a new alert with timestamp"""
    current_time = time.time()
    st.session_state.alerts.append({"text": alert_text, "timestamp": current_time})
    # Immediately play sound when alert is added
    play_alert_sound()

def update_alerts():
    """Update alerts list and remove expired alerts"""
    current_time = time.time()
    st.session_state.alerts = [alert for alert in st.session_state.alerts 
                              if current_time - alert["timestamp"] <= ALERT_DURATION]
    
    if st.session_state.alerts:
        alerts_html = '<div class="alert-container">'
        for alert in st.session_state.alerts:
            alerts_html += f"<p>{alert['text']}</p>"
        alerts_html += '</div>'
        alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)
    else:
        alerts_placeholder.empty()

was_yawning = False

# Main loop
try:
    while True:
        frame = vs.read()
        if frame is None:
            st.error("Error: Could not read frame. Check if the camera is connected.")
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        # Update alerts at the start of each loop iteration
        update_alerts()

        # Detect faces
        rects = detector(gray, 0)

        # Process each face
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Eye aspect ratio
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Mouth aspect ratio
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            # Head pose
            image_points = np.array([
                shape[33],  # Nose tip
                shape[8],   # Chin
                shape[36],  # Left eye left corner
                shape[45],  # Right eye right corner
                shape[48],  # Left Mouth corner
                shape[54]   # Right mouth corner
            ], dtype="double")

            head_tilt_degree, _, _, _ = getHeadTiltAndCoords(size, image_points, frame.shape[0])
            tilt_angle = abs(float(head_tilt_degree[0])) if head_tilt_degree else 0

            # Update metrics with labels and values
            ear_metric.metric("Eye Aspect Ratio", f"{ear:.2f}", f"Threshold: {EYE_AR_THRESH}")
            mar_metric.metric("Mouth Aspect Ratio", f"{mar:.2f}", f"Threshold: {MOUTH_AR_THRESH}")
            tilt_metric.metric("Head Tilt Angle", f"{tilt_angle:.1f}°", f"Threshold: {HEAD_TILT_THRESH}°")

            # Add status indicators
            if ear < EYE_AR_THRESH:
                ear_metric.markdown(":red_circle: Eyes Closing", unsafe_allow_html=True)
            else:
                ear_metric.markdown(":green_circle: Eyes Open", unsafe_allow_html=True)

            if mar > MOUTH_AR_THRESH:
                mar_metric.markdown(":warning: Yawning", unsafe_allow_html=True)
            else:
                mar_metric.markdown(":white_check_mark: Normal", unsafe_allow_html=True)

            if tilt_angle > HEAD_TILT_THRESH:
                tilt_metric.markdown(":warning: Head Tilted", unsafe_allow_html=True)
            else:
                tilt_metric.markdown(":white_check_mark: Normal", unsafe_allow_html=True)

            # Process alerts
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                LONG_COUNTER += 1
                
                if LONG_COUNTER >= LONG_CLOSED_FRAMES:
                    if LONG_COUNTER == LONG_CLOSED_FRAMES:  # Just triggered
                        add_alert("DROWSINESS ALERT! Eyes Closed Too Long!")
                        eyes_closed_alert_time = reset_alert_counters("eyes_closed")
                
                if COUNTER == EYE_AR_CONSEC_FRAMES:
                    blink_times.append(time.time())
                    recent_blinks = check_time_window(blink_times)
                    if recent_blinks >= 5:  # 5 blinks in 30 seconds
                        if blink_condition_met_time is None:
                            blink_condition_met_time = time.time()
                        if time.time() - blink_condition_met_time >= ALERT_DELAY:
                            if not blink_alert_played:
                                add_alert("DROWSINESS ALERT! Frequent Blinking!")
                                blink_alert_played = True
                                blink_alert_time = reset_alert_counters("blink")
                                blink_condition_met_time = None
                    else:
                        blink_condition_met_time = None
            else:
                COUNTER = 0
                LONG_COUNTER = 0
                blink_alert_played = False

            # Check for yawning
            if mar > MOUTH_AR_THRESH:
                if not was_yawning:
                    yawn_times.append(time.time())
                    recent_yawns = check_time_window(yawn_times)
                    if recent_yawns >= 3:  # 3 yawns in 30 seconds
                        if yawn_condition_met_time is None:
                            yawn_condition_met_time = time.time()
                        if time.time() - yawn_condition_met_time >= ALERT_DELAY:
                            if not yawn_alert_played:
                                add_alert("DROWSINESS ALERT! Frequent Yawning!")
                                yawn_alert_played = True
                                yawn_alert_time = reset_alert_counters("yawn")
                                yawn_condition_met_time = None
                    else:
                        yawn_condition_met_time = None
                was_yawning = True
            else:
                was_yawning = False
                yawn_alert_played = False

            # Head tilt alert
            if tilt_angle > HEAD_TILT_THRESH:
                CONTINUOUS_HEAD_TILT_COUNTER += 1
                
                if CONTINUOUS_HEAD_TILT_COUNTER >= HEAD_TILT_CONSEC_FRAMES:
                    if CONTINUOUS_HEAD_TILT_COUNTER == HEAD_TILT_CONSEC_FRAMES:  # Just triggered
                        add_alert("DROWSINESS ALERT! Head Tilt Too Long!")
                        head_tilt_too_long_alert_time = reset_alert_counters("head_tilt_long")
        
                if CONTINUOUS_HEAD_TILT_COUNTER == 1:
                    head_tilt_times.append(time.time())
                    recent_tilts = check_time_window(head_tilt_times)
                    if recent_tilts >= 5:
                        if head_tilt_condition_met_time is None:
                            head_tilt_condition_met_time = time.time()
                        if time.time() - head_tilt_condition_met_time >= ALERT_DELAY:
                            if not head_tilt_alert_played:
                                add_alert("DROWSINESS ALERT! Frequent Head Tilting!")
                                head_tilt_alert_played = True
                                head_tilt_alert_time = reset_alert_counters("head_tilt")
                                head_tilt_condition_met_time = None
                    else:
                        head_tilt_condition_met_time = None
            else:
                CONTINUOUS_HEAD_TILT_COUNTER = 0
                head_tilt_alert_played = False

            # Draw face bounding box
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

                # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Draw mouth contour
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # Convert frame for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame with debug info
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tilt: {tilt_angle:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update video feed
            video_placeholder.image(frame)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
finally:
    vs.stop()
    cv2.destroyAllWindows()

# Add custom CSS for styling
st.markdown("""
<style>
    /* Hide audio element container */
    [data-testid="stAudio"] {
        display: none;
    }
    
    /* Hide the HTML components height */
    iframe {
        height: 0px !important;
    }
</style>
""", unsafe_allow_html=True)