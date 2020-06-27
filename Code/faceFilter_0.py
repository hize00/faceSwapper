import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from scipy.interpolate import interp1d
from imutils import face_utils
import argparse
import os
from PIL import Image
import dlib
import time
import random
#from Code import config
import config
#from Code import faceSwap
import faceSwap


"""GLOBAL VARIABLES"""
appName = config.APP_NAME

landmarks_file = config.FACE_LANDMARK_FILE

FRONT_HEIGHT = config.FRONT_BOX_PIXEL_HEIGHT

alpha = config.ALPHA_OVERLAY

REPLY_LOGOS_FOLDER = config.REPLY_LOGO_FOLDER
REPLY_LOGOS = []
for filename in os.listdir(REPLY_LOGOS_FOLDER):
    if filename.endswith("png") or filename.endswith("jpg"):
        filepath = os.path.join(REPLY_LOGOS_FOLDER, filename)
        REPLY_LOGOS.append(filepath)

DRUGS_FACES_FOLDER = config.ADDICTS_FACES_FOLDER
DRUGS_FACES = config.ADDICT_FACES_DICT

faceTestImg = "D:\\RANDOM CLF\\faceFilters\\Code\\faces\\blonde.jpg"
# based on 68 points of shape_predictor
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


def getEyeLandmarkPts(face_landmark_points):
    '''
    Input: Coordinates of Bounding Box single face
    Returns: eye's landmark points
    '''
    face_landmark_points[36][0] -= 5
    face_landmark_points[39][0] += 5
    face_landmark_points[42][0] -= 5
    face_landmark_points[45][0] += 5
    L_eye_top = face_landmark_points[36: 40]
    L_eye_bottom = np.append(face_landmark_points[39: 42], face_landmark_points[36]).reshape(4, 2)
    R_eye_top = face_landmark_points[42:  46]
    R_eye_bottom = np.append(face_landmark_points[45:48], face_landmark_points[42]).reshape(4, 2)
    return [L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom]


def getBrowsLandmarkPts(face_landmark_points):
    L_brow = face_landmark_points[17:22]
    R_brow = face_landmark_points[22:26]
    return [L_brow, R_brow]


def getFrontLandmarkPts(face_landmark_points):
    front_points = np.concatenate((face_landmark_points[17:22], face_landmark_points[22:27]))
    # increment y
    for i in range(0, len(front_points)):
        front_points[i][1] -= FRONT_HEIGHT
    return front_points


def interpolateCoordinates(xy_coords, x_intrp):
    x = xy_coords[:, 0]
    y = xy_coords[:, 1]
    intrp = interp1d(x, y, kind='quadratic')
    y_intrp = intrp(x_intrp)
    y_intrp = np.floor(y_intrp).astype(int)
    return y_intrp


def getEyelinerPoints(eye_landmark_points):
    '''
    Takes an array of eye coordinates and interpolates them:
    '''
    L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom = eye_landmark_points
    L_interp_x = np.arange(L_eye_top[0][0], L_eye_top[-1][0], 1)
    R_interp_x = np.arange(R_eye_top[0][0], R_eye_top[-1][0], 1)

    L_interp_top_y = interpolateCoordinates(L_eye_top, L_interp_x)
    L_interp_bottom_y = interpolateCoordinates(L_eye_bottom, L_interp_x)
    R_interp_top_y = interpolateCoordinates(R_eye_top, R_interp_x)
    R_interp_bottom_y = interpolateCoordinates(R_eye_bottom, R_interp_x)

    return [(L_interp_x, L_interp_top_y, L_interp_bottom_y), (R_interp_x, R_interp_top_y, R_interp_bottom_y)]


def getBrowslinerPoints(brows_landmark_points):
    L_brow, R_brow = brows_landmark_points

    L_interp_x = np.arange(L_brow[0][0], L_brow[-1][0], 1)
    R_interp_x = np.arange(R_brow[0][0], R_brow[-1][0], 1)
    L_interp_y = interpolateCoordinates(L_brow, L_interp_x)
    R_interp_y = interpolateCoordinates(R_brow, R_interp_x)

    return [(L_interp_x, L_interp_y), (R_interp_x, R_interp_y)]


def drawEyeliner(img, interp_pts):
    L_eye_interp, R_eye_interp = interp_pts

    L_interp_x, L_interp_top_y, L_interp_bottom_y = L_eye_interp
    R_interp_x, R_interp_top_y, R_interp_bottom_y = R_eye_interp

    overlay = img.copy()
    # overlay = np.empty(img.shape)
    # overlay = np.zeros_like(img)

    for i in range(len(L_interp_x) - 2):
        x1 = L_interp_x[i]
        y1_top = L_interp_top_y[i]
        x2 = L_interp_x[i + 1]
        y2_top = L_interp_top_y[i + 1]
        cv2.line(overlay, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = L_interp_bottom_y[i]
        y2_bottom = L_interp_bottom_y[i + 1]
        cv2.line(overlay, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    for i in range(len(R_interp_x) - 2):
        x1 = R_interp_x[i]
        y1_top = R_interp_top_y[i]
        x2 = R_interp_x[i + 1]
        y2_top = R_interp_top_y[i + 1]
        cv2.line(overlay, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = R_interp_bottom_y[i]
        y2_bottom = R_interp_bottom_y[i + 1]
        cv2.line(overlay, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    # background = Image.fromarray(img) # .convert("1")
    # foreground = Image.fromarray(overlay).convert("1")
    # newImg = Image.composite(foreground, background, foreground)#, mask='1')
    # # img = cv2.bitwise_and(overlay, img)
    # return cv2.cvtColor(np.array(newImg), cv2.COLOR_RGB2BGR)
    return overlay


def drawBrowliner(img, inter_pts):
    L_brow_interp, R_brow_interp = inter_pts
    L_interp_x, L_interp_y = L_brow_interp
    R_interp_x, R_interp_y = R_brow_interp

    overlay = img.copy()

    for i in range(len(L_interp_x) - 2):
        x1 = L_interp_x[i]
        y1 = L_interp_y[i]
        x2 = L_interp_x[i + 1]
        y2 = L_interp_y[i + 1]
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
    for i in range(len(R_interp_x) - 2):
        x1 = R_interp_x[i]
        y1 = R_interp_y[i]
        x2 = R_interp_x[i + 1]
        y2 = R_interp_y[i + 1]
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

    return overlay


"""
def ExtractFace(frame):
    path_output = "D:\\RANDOM CLF\\faceFilters\\Code\\faces\\output\\"
    offset = 50
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)
    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            # two points of the bounding box
            x1, y1, x2, y2, w, h = bb.left(), bb.top(), bb.right() + 1, bb.bottom() + 1, bb.width(), bb.height()
            # cv2.rectangle(frame, (x1, y1), (x2+w, y2+h), (255, 0, 0), 2)
            # face_rectangle = frame[y1:y1 + h, x1:x1 + w]
            face_rectangle = frame[y1-offset: y1+h+offset, x1-offset: x1+w+offset]
            print("[INFO] Object found. Saving clean face.")
            cv2.imwrite(path_output + str(w) + str(h) + '_faces.jpg', face_rectangle)
            swappedFace = faceSwap.faceSwap(face_rectangle, faceImg)
            frame[y1 - offset: y1 + h + offset, x1 - offset: x1 + w + offset] = swappedFace
            print("[INFO] Object found. Saving swapped face.")
            cv2.imwrite(path_output + str(w) + str(h) + '_faces.jpg', frame)
            # time.sleep(2)
        return face_rectangle
    else:
        return frame
"""


def Eyeliner(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The 2nd argument means that we upscale the image by 'x' number of times to detect more faces.
    bounding_boxes = face_detector(gray, 0)
    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            # two points of the bounding box
            face_landmark_points = lndMrkDetector(gray, bb)
            # mapped to face_landmark_68 points
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            eye_landmark_points = getEyeLandmarkPts(face_landmark_points)
            eyeliner_points = getEyelinerPoints(eye_landmark_points)
            op = drawEyeliner(frame, eyeliner_points)
        return op
    else:
        return frame


def Browliner(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)
    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            brow_landmark_points = getBrowsLandmarkPts(face_landmark_points)
            browliner_points = getBrowslinerPoints(brow_landmark_points)
            op = drawBrowliner(frame, browliner_points)
        return op
    else:
        return frame


def ReplyDrawer(frame, replyImg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)
    foreground = cv2.imread(replyImg)
    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            front_landmark_points = getFrontLandmarkPts(face_landmark_points)
            bottom_x = front_landmark_points[0][0]
            top_y = front_landmark_points[-1][1] + 50
            clean_frame = frame.copy()
            frame[top_y:top_y + foreground.shape[0], bottom_x:bottom_x + foreground.shape[1]] = foreground
            frame = clean_frame.copy()
        return frame
    else:
        return frame


def WordWriter(frame, words):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)
    offset_step = 5
    offset = len(words)*offset_step
    # TODO: better font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 204, 0)
    lineType = 2

    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            front_landmark_points = getFrontLandmarkPts(face_landmark_points)
            left_x = front_landmark_points[0][0] + 25 - offset
            left_y = front_landmark_points[0][1] + 115
            bottomLeftCornerOfText = (left_x, left_y)
            clean_frame = frame.copy()
            cv2.putText(frame, words, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            frame = clean_frame.copy()
    return


def FaceSwapper(frame, faceImg, words):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)
    offset = 50
    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            x1, y1, x2, y2, w, h = bb.left(), bb.top(), bb.right() + 1, bb.bottom() + 1, bb.width(), bb.height()
            face_rectangle = frame[y1 - offset: y1 + h + offset, x1 - offset: x1 + w + offset]
            swappedFace = faceSwap.faceSwap(face_rectangle, faceImg)
            frame[y1 - offset: y1 + h + offset, x1 - offset: x1 + w + offset] = swappedFace
            if len(words) > 0:
                WordWriter(frame, words)
        return frame
    else:
        return frame


""" ========================================================
--->> CALL FILTER FUNCTIONS
======================================================== """


def video(src=0, face_filter='eyeliner'):
    counter_timeout = 0
    cap = cv2.VideoCapture(src)
    if args['save']:
        if os.path.isfile(args['save'] + '.avi'):
            os.remove(args['save'] + '.avi')
        output_video = cv2.VideoWriter(args['save'] + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened:
        open_flag, frame = cap.read()
        """ ========================================================
        --->> BROWLINER
        ======================================================== """
        if face_filter.lower() == 'browliner':
            output_frame = Browliner(frame)

            """ ====================================================
            --->> REPLYER
            ======================================================== 
            """
        elif face_filter.lower() == 'reply':
            if counter_timeout < config.TIMEOUT_QUESTION:
                WordWriter(frame, "Which Reply Are You?")
                output_frame = frame
            else:
                current_img = random.choice(REPLY_LOGOS)
                if counter_timeout >= config.TIMEOUT_MAX:
                    ReplyDrawer(frame, final_img)
                elif counter_timeout % config.TIMEOUT_ITERATION == 0:
                    # change current_img
                    current_img = random.choice(REPLY_LOGOS)
                    ReplyDrawer(frame, current_img)
                    final_img = current_img
                else:
                    ReplyDrawer(frame, current_img)
                output_frame = frame
            counter_timeout += 1

            """ ====================================================
            --->> WHICH DRUG
            ======================================================== 
            """
        elif face_filter.lower() == 'drug':
            if counter_timeout < config.TIMEOUT_QUESTION:
                WordWriter(frame, "Which Drug Are You?")
                output_frame = frame
            else:
                dict_entry = random.choice(DRUGS_FACES)
                current_img = dict_entry['path']
                current_drug = dict_entry['drug']
                if counter_timeout >= config.TIMEOUT_MAX:
                    output_frame = FaceSwapper(frame, final_img, final_drug)
                elif counter_timeout % config.TIMEOUT_ITERATION == 0:
                    # change current_img
                    dict_entry = random.choice(DRUGS_FACES)
                    current_img = dict_entry['path']
                    current_drug = dict_entry['drug']
                    # output_frame = FaceSwapper(frame, current_img, '')
                    WordWriter(frame, current_drug)
                    final_img = current_img
                    final_drug = current_drug
                else:
                    # output_frame = FaceSwapper(frame, current_img, '')
                    WordWriter(frame, current_drug)
                output_frame = frame
            counter_timeout += 1

            """ ====================================================
            --->> STRONZO METER
            ======================================================== 
            """
        elif face_filter.lower() == 'stronzo':
            percentages = list(range(101))
            if counter_timeout < config.TIMEOUT_QUESTION:
                WordWriter(frame, "Quanto sei stronzo?")
                output_frame = frame
            else:
                current_percentage = random.choice(percentages)
                if counter_timeout >= config.TIMEOUT_MAX:
                    WordWriter(frame, str(final_percentage) + '%')
                elif counter_timeout % config.TIMEOUT_ITERATION == 0:
                    current_percentage = random.choice(percentages)
                    WordWriter(frame, str(current_percentage) + '%')
                    final_percentage = current_percentage
                else:
                    WordWriter(frame, str(current_percentage) + '%')
                output_frame = frame
            counter_timeout += 1

            """
            DEFAULT: EYELINER
            """
        else:
            output_frame = Eyeliner(frame)

        if args['save']:
            output_video.write(output_frame)
        cv2.imshow(appName, cv2.resize(output_frame, (600, 600)))
        # close webcam with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args['save']:
        output_video.release()

    cap.release()
    cv2.destroyAllWindows()


def image(source, face_filter='eyeliner'):
    if os.path.isfile(source):
        img = cv2.imread(source)

        # CALL FILTER FUNCTIONS
        if face_filter.lower() == 'browliner':
            output_frame = Browliner(img)
        else:
            output_frame = Eyeliner(img)

        cv2.imshow(appName, cv2.resize(output_frame, (600, 600)))
        if args['save']:
            if os.path.isfile(args['save'] + '.png'):
                os.remove(args['save'] + '.png')
            cv2.imwrite(args['save'] + '.png', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("File not found :( ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filter", required=False, help="Type of filter to apply on the face")
    ap.add_argument("-v", "--video", required=False, help="Path to video file")
    ap.add_argument("-i", "--image", required=False, help="Path to image")
    ap.add_argument("-d", "--dat", required=False, help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-t", "--thickness", required=False, help="Enter int value of thickness (recommended 0-5)")
    ap.add_argument("-c", "--color", required=False, help='Enter R G B color value', nargs=3)
    ap.add_argument("-s", "--save", required=False, help='Enter the file name to save')
    args = vars(ap.parse_args())

    if args['dat']:
        dataFile = args['dat']

    else:
        dataFile = landmarks_file

    color = (0, 0, 0)
    thickness = 2
    filter = "Eyeliner"
    face_detector = dlib.get_frontal_face_detector()
    lndMrkDetector = dlib.shape_predictor(dataFile)

    if args['filter']:
        filter = args['filter']

    if args['color']:
        color = list(map(int, args['color']))
        color = tuple(color)

    if args['thickness']:
        thickness = int(args['thickness'])

    if args['image']:
        image(args['image'])

    if args['video'] and args['video'] != 'webcam':
        if os.path.isfile(args['video']):
            video(args['video'])
        else:
            print("File not found :( ")

    elif args['video'] == 'webcam':
        video(0, filter)


    # TEST RUN FROM PYCHARM
    # filter = "stronzo"
    # video(0, filter)
    print("Done.")
