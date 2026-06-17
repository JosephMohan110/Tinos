import cv2
import argparse
import sys
import logging
from typing import Optional

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FaceDetector:

    
    def __init__(self, cascade_path: Optional[str] = None):

        # Default to OpenCV's built-in frontal face cascade if none is provided
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logging.error(f"Could not load Haar cascade from {cascade_path}")
            sys.exit(1)
            
        # UI Constants
        self.box_color = (255, 0, 0)  # Blue bounding box
        self.text_color = (255, 0, 0) # Blue text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        

        self.scale_factor = 1.1
        self.min_neighbors = 6  # Increased from 5 to 6 to be stricter on "only faces"
        self.min_size = (30, 30)
            
    def process_frame(self, frame):

        # Convert to grayscale for Haar Cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with tuned parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors, 
            minSize=self.min_size
        )
        
        # Draw rectangles and labels around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.box_color, 2)
            cv2.putText(frame, 'Face', (x, y-10), self.font, 0.9, self.text_color, 2)
            
        return frame, len(faces)

    def detect_in_image(self, image_path: str):

        logging.info(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        
        if img is None:
            logging.error(f"Could not read image at {image_path}. Please check the path.")
            sys.exit(1)
            
        processed_img, face_count = self.process_frame(img)
        logging.info(f"Successfully detected {face_count} face(s).")
        
        cv2.imshow('Face Detection - Image', processed_img)
        logging.info("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_in_webcam(self):
 
        logging.info("Initializing webcam capture...")
        # Use DirectShow backend for better compatibility on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            logging.error("Could not open webcam. Ensure it's not being used by another application.")
            sys.exit(1)
            
        logging.info("Webcam opened successfully. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to read frame from webcam.")
                    break
                    
                processed_frame, _ = self.process_frame(frame)
                
                cv2.imshow('Face Detection - Webcam', processed_frame)
                
                # Check for 'q' key press to exit 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit requested by user. Closing...")
                    break
        finally:
            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Professional Face Detection System using OpenCV")
    parser.add_argument(
        "-i", "--image", 
        type=str, 
        help="Optional path to an input image. If not provided, the system defaults to the webcam."
    )
    args = parser.parse_args()
    
    # Initialize the detector
    detector = FaceDetector()
    
    # Route to the appropriate detection method based on arguments
    if args.image:
        detector.detect_in_image(args.image)
    else:
        detector.detect_in_webcam()

if __name__ == "__main__":
    main()
