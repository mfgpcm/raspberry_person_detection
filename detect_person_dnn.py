"""MIT License

Copyright (c) 2024 Peter Munk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """

import argparse
import json
import os
import logging
import time
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import smtplib
from email.message import EmailMessage

def send_email(video_path: str, sender_email: str, receiver_email: str, smtp_url: str, smtp_port: int, password: str):
    msg = EmailMessage()
    msg['Subject'] = 'Person Detection Alert'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('A person was detected in the video stream.')

    try:
        with open(video_path, 'rb') as f:
            file_data = f.read()
            file_name = f.name
            msg.add_attachment(file_data, maintype='video', subtype='mp4', filename=file_name)

        with smtplib.SMTP_SSL(smtp_url, smtp_port) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)
            logging.info("Mail sent with attached %s", video_path)
    except:
        logging.exception("Sending mail failed")


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (10 + bbox.origin_x, 20 + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    return image

def open_cam(rtsp_url: str, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    # Start capturing video input from the camera
    
    ## From https://lindevs.com/capture-rtsp-stream-from-ip-camera-using-opencv/
    ## Leads to many decoding errors
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp' 
    try:
        cap = cv2.VideoCapture(rtsp_url)
    except:
        logging.error("Could not open cam %s", rtsp_url)
    
    ## DEBUG
    #cap = cv2.VideoCapture('recordings/test.mp4')

    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run(model: str, recs: str, rtsp_url: str, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, period: int, min_person_frames: int, min_no_person_frames: int,
        interactive: bool, min_conf: float, send_mail: bool, sender_email: str, 
        receiver_email: str, smtp_url: str, smtp_port: int, password: str) -> None:

    cap = open_cam(rtsp_url=rtsp_url, width=width, height=height)
        
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = None
    
    # Initialize frame counters
    frames_with_person = 0
    frames_without_person = 0

    sleep_if_failed = 1

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=5, score_threshold=min_conf)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    logging.info("%s model loaded, entering main loop", model)

    # Variables to calculate FPS
    counter = 0 
    fps = 0
    calc_fps_ever = 5
    start_time = time.time()

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logging.error("Unable to read from webcam. Waiting for %i seconds until retry", sleep_if_failed)
            try:
                cap.release()
            except:
                logging.error("Failed to release cam")
            while (not cap) or (not cap.isOpened()):
                time.sleep(sleep_if_failed)
                cap = open_cam(rtsp_url=rtsp_url, width=width, height=height)
                sleep_if_failed *= 5
            logging.info("Reconnected to %s", rtsp_url)           
        else:
            sleep_if_failed = 1
            counter += 1
                        
            image = cv2.resize(image, (width, height))

            # Convert the image from BGR to RGB as required by the TFLite model.
            #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(image)

            # Run object detection estimation using the model.
            detection_result = detector.detect(input_tensor)

            person_detected = False
            for detection in detection_result.detections:
                for category in detection.categories:
                    if category.category_name == "person":
                        if category.score > min_conf:
                            person_detected = True
                            logging.info("Person detected with confidence {:.2f}".format(category.score))
                            break

            if person_detected:
                if frames_with_person == 0:
                    try:
                        video_dir = os.path.join(recs, time.strftime('%Y-%m-%d'))
                        if not os.path.isdir(video_dir):
                            os.mkdir(video_dir)
                        video_path = os.path.join(video_dir, 'detection_'+time.strftime('%Y-%m-%d_%H-%M-%S')+".webm")
                        try:
                            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        except:
                            logging.exception()
                        logging.debug("Started recording %s", video_path)
                    except:
                        logging.exception()
                frames_with_person += 1
                frames_without_person = 0
            else:
                if frames_with_person > 0:
                    frames_without_person += 1

            # Record the frame if we have started detection
            if frames_with_person > 0:
                out.write(image)

            # If a person was detected for at least min_person_frames frames and then not detected for 4 frames, send the email
            if frames_with_person >= min_person_frames and frames_without_person >= min_no_person_frames:
                logging.info("Person detected for %i frames and not detected since %i frames", frames_with_person, frames_without_person)
                try:
                    out.release()
                    logging.debug("Video saved here %s", video_path)
                except:
                    logging.exception()
                if send_mail:
                    send_email(video_path, sender_email, receiver_email, smtp_url, smtp_port, password)
                frames_with_person = 0
                frames_without_person = 0
                out = None
            
            if interactive:
                # Stop the program if the ESC key is pressed.
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                # Draw keypoints and edges on input image
                image = visualize(image, detection_result)
                cv2.imshow('object_detector', image)

            # Calculate remaining time to the next 500 ms interval
            elapsed_time = (time.time() - start_time) * 1000
            remaining_time = max(1, period - int(elapsed_time) % period)
            # Wait for the remainder of the 500 ms period
            time.sleep(remaining_time/1000)

            # Calculate the FPS
            if time.time() - start_time >= calc_fps_ever:
                fps = counter / (time.time() - start_time)
                start_time = time.time()
                counter = 0
                logging.debug("current fps: {:.1f} (opencv = {:.1f})".format(fps, cap.get(cv2.CAP_PROP_FPS)))



    cap.release()
    if interactive:
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_file', 
        help='Path to JSON config file', 
        required=False,
        default='detect_person_config.json')
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        # best tflite model https://www.ejtech.io/learn/tflite-object-detection-model-comparison
        # https://www.kaggle.com/models?framework=tfLite&query=image-object-detection&tfhub-redirect=true
        default='model/ssd-mobilenet-v1-tflite-metadata-v2.tflite')
    parser.add_argument(
        '--recs',
        help='Path of the folder where recordings are stored.',
        required=False,
        default='./recordings')
    parser.add_argument(
        '--cameraUrl', 
        help='Url of camera.', 
        required=False, 
        #https://www.ispyconnect.com/camera/lupus
        )
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=420)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=2)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--min_person_frames',
        help='Number of frames a person must be detected (recordings starts with first frame) to send a video',
        required=False,
        type=int,
        default=1)
    parser.add_argument(
        '--min_no_person_frames',
        help='Number of frames no person is detected aynmore to send the video',
        required=False,
        type=int,
        default=20)
    parser.add_argument(
        '--interactive',
        help='Whether to run with a window (exit with ESC)',
        action='store_true',
        required=False,
        default=False)   
    parser.add_argument(
        '--min_conf',
        help='Minimum confidence for a detected person',
        required=False,
        type=float,
        default=0.4)
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '-s', '--send_mail',
        help="Send an email",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        '--receiver_email', 
        help='Receiver email adress',
        required=False, 
        )
    parser.add_argument(
        '--sender_email', 
        help='Sender email adress and login',
        required=False, 
        )
    parser.add_argument(
        '--smtp_url', 
        help='SMTP server url',
        default="smtp.gmail.com",
        required=False, 
        )
    parser.add_argument(
        '--password', 
        help='Sender passowrd',
        required=False, 
        )
    parser.add_argument(
        '--smtp_port',
        help='SMTP server port',
        required=False,
        type=int,
        default=465)
    parser.add_argument(
        '--period',
        help='Timing period of reading the images in s',
        required=False,
        type=int,
        default=500)    
    args = parser.parse_args()    

    # Read config from file if specified
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_settings = json.load(f)
            # Overwrite any possible given arguments
            for key, value in config_settings.items():
                setattr(args, key, value)    
    
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    
    if args.loglevel == logging.DEBUG:
        logging.debug(args)

    if not os.path.isdir(args.recs):
        logging.error("Recordings directory does not exist")
    else: 
        run(args.model, args.recs, args.cameraUrl, args.frameWidth, args.frameHeight, int(args.numThreads),
            bool(args.enableEdgeTPU), int(args.period), int(args.min_person_frames), int(args.min_no_person_frames),
            bool(args.interactive), float(args.min_conf), bool(args.send_mail),
            args.sender_email, args.receiver_email, args.smtp_url, int(args.smtp_port), args.password)


if __name__ == '__main__':
  main()