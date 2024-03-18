from flask import Flask, request, jsonify
from flask_cors import CORS  # Ensure this import is correct
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Flask server is running!"


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    # Convert base64 image to NumPy array
    img_str = request.form['image'].split(',')[1]
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Parameters for chessboard calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboardSize = (6, 9)  # Define as (columns-1, rows-1)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        response = {
            "status": "success",
            "mtx": mtx.tolist(),  # Camera matrix
            "dist": dist.tolist(),  # Distortion coefficients
            # Consider converting rvecs and tvecs if needed
        }
    else:
        response = {"status": "failure",
                    "message": "Chessboard not found. Ensure the entire chessboard is visible in the frame."}

    return jsonify(response)


if __name__ == '__main__':
    app.run()
