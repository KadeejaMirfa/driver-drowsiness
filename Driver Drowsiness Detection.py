#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from playsound import playsound
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# Initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# Frame dimensions
frame_width = 1024
frame_height = 576

# Constants for drowsiness detection
EYE_AR_THRESH = 0.20
MOUTH_AR_THRESH = 0.79
HEAD_TILT_THRESH = 22
EYE_AR_CONSEC_FRAMES = 3
LONG_CLOSED_FRAMES = 150  # 5 seconds at 30 fps
HEAD_TILT_CONSEC_FRAMES = 90  # 3 seconds at 30 fps

# Define facial landmarks for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize image_points for head pose estimation
image_points = np.array([
    (0, 0),     # Nose tip 34
    (0, 0),     # Chin 9
    (0, 0),     # Left eye left corner 37
    (0, 0),     # Right eye right corner 46
    (0, 0),     # Left Mouth corner 49
    (0, 0)      # Right mouth corner 55
], dtype="double")

# Counters and time tracking
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

def display_alert(frame, alert_text, position_y):
    """Display an alert text on the frame at the specified vertical position"""
    cv2.putText(frame, alert_text, (10, position_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

was_yawning = False

while True:
    # Grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame. Check if the camera is connected and accessible.")
        exit()

    frame = imutils.resize(frame, width=frame_width, height=frame_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_time = time.time()

    # Loop over the face detections
    for rect in rects:
        # Compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            LONG_COUNTER += 1
            
            # Check for long eye closure (5 seconds)
            if LONG_COUNTER >= LONG_CLOSED_FRAMES:
                display_alert(frame, "DROWSINESS ALERT! Eyes Closed Too Long!", 50)
                if LONG_COUNTER == LONG_CLOSED_FRAMES:  # Just triggered
                    playsound('beep.wav')  # Play beep sound for head tilt too long
                    eyes_closed_alert_time = reset_alert_counters("eyes_closed")
                    
            # Check for multiple short eye closures
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Only count as a new blink if we just crossed the threshold
                if COUNTER == EYE_AR_CONSEC_FRAMES:
                    blink_times.append(time.time())
                    recent_blinks = check_time_window(blink_times)
                    if recent_blinks >= 5:  # 5 blinks in 30 seconds
                        # Check if this is the first time condition is met
                        if blink_condition_met_time is None:
                            blink_condition_met_time = time.time()
                        
                        # If 2 seconds have passed since condition was met
                        if time.time() - blink_condition_met_time >= ALERT_DELAY:
                            display_alert(frame, "DROWSINESS ALERT! Frequent Blinking!", 80)
                            if not blink_alert_played:
                                playsound('beep.wav')  # Play beep sound for frequent blinking
                                blink_alert_played = True
                                blink_alert_time = reset_alert_counters("blink")
                                blink_condition_met_time = None  # Reset the timer
                    else:
                        blink_condition_met_time = None  # Reset if conditions are no longer met
        else:
            COUNTER = 0
            LONG_COUNTER = 0
            blink_alert_played = False  # Reset the alert flag

        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        
        # Compute the convex hull for the mouth, then visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for yawning
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Only count as a new yawn if we weren't yawning in the previous frame
            if not was_yawning:
                yawn_times.append(time.time())
                recent_yawns = check_time_window(yawn_times)
                if recent_yawns >= 3:  # 3 yawns in 30 seconds
                    # Check if this is the first time condition is met
                    if yawn_condition_met_time is None:
                        yawn_condition_met_time = time.time()
                    
                    # If 2 seconds have passed since condition was met
                    if time.time() - yawn_condition_met_time >= ALERT_DELAY:
                        display_alert(frame, "DROWSINESS ALERT! Frequent Yawning!", 110)
                        if not yawn_alert_played:
                            playsound('beep.wav')  # Play beep sound for frequent yawning
                            yawn_alert_played = True
                            yawn_alert_time = reset_alert_counters("yawn")
                            yawn_condition_met_time = None  # Reset the timer
                else:
                    yawn_condition_met_time = None  # Reset if conditions are no longer met
            was_yawning = True
        else:
            was_yawning = False
            yawn_alert_played = False  # Reset the alert flag

        # Loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Draw the determinant image points onto the person's face
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            tilt_angle = abs(float(head_tilt_degree[0]))
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Check for head tilt above threshold
            if tilt_angle > HEAD_TILT_THRESH:
                CONTINUOUS_HEAD_TILT_COUNTER += 1
                
                # Check for continuous head tilt
                if CONTINUOUS_HEAD_TILT_COUNTER >= HEAD_TILT_CONSEC_FRAMES:
                    display_alert(frame, "DROWSINESS ALERT! Head Tilt Too Long!", 140)
                    if CONTINUOUS_HEAD_TILT_COUNTER == HEAD_TILT_CONSEC_FRAMES:  # Just triggered
                        playsound('beep.wav')  # Play beep sound for head tilt too long
                        head_tilt_too_long_alert_time = reset_alert_counters("head_tilt_long")
        
                # Check for multiple head tilts
                if CONTINUOUS_HEAD_TILT_COUNTER == 1:  # Just started tilting
                    head_tilt_times.append(time.time())
                    recent_tilts = check_time_window(head_tilt_times)
                    if recent_tilts >= 5:  # 5 tilts in 30 seconds
                        # Check if this is the first time condition is met
                        if head_tilt_condition_met_time is None:
                            head_tilt_condition_met_time = time.time()
                        
                        # If 2 seconds have passed since condition was met
                        if time.time() - head_tilt_condition_met_time >= ALERT_DELAY:
                            display_alert(frame, "DROWSINESS ALERT! Frequent Head Tilting!", 170)
                            if not head_tilt_alert_played:
                                playsound('beep.wav')  # Play beep sound for frequent head tilts
                                head_tilt_alert_played = True
                                head_tilt_alert_time = reset_alert_counters("head_tilt")
                                head_tilt_condition_met_time = None  # Reset the timer
                    else:
                        head_tilt_condition_met_time = None  # Reset if conditions are no longer met
            else:
                CONTINUOUS_HEAD_TILT_COUNTER = 0
                head_tilt_alert_played = False  # Reset the alert flag

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()