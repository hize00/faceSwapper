import os

APP_NAME = "FaceFilter"
FACE_LANDMARK_FILE = "shape_predictor_68_face_landmarks.dat"
ALPHA_OVERLAY = 0.1
FRONT_BOX_PIXEL_HEIGHT = 150
REPLY_LOGO_FOLDER = "D:\\RANDOM CLF\\faceFilters\\Code\\replyLogos"
ADDICTS_FACES_FOLDER = "D:\\RANDOM CLF\\faceFilters\\Code\\faces\\addicts"
TIMEOUT_QUESTION = 125
TIMEOUT_ITERATION = 25
TIMEOUT_MAX = 200
OUTPUT_FOLDER = "D:\\RANDOM CLF\\faceFilters\\Code\\output"

ADDICT_FACES_DICT = []
for filename in os.listdir(ADDICTS_FACES_FOLDER):
    if filename.endswith("png") or filename.endswith("jpg"):
        dict = {}
        filepath = os.path.join(ADDICTS_FACES_FOLDER, filename)
        dict['path'] = filepath
        if 'marley' in filepath:
            dict['drug'] = 'WEED'
        elif 'stanley' in filepath:
            dict['drug'] = 'ALCOHOL'
        elif 'escobar' in filepath:
            dict['drug'] = 'COCAINE'
        elif 'lohan' in filepath:
            dict['drug'] = 'METH'
        elif 'maradona' in filepath:
            dict['drug'] = 'COCAINE'
        elif 'vicious' in filepath:
            dict['drug'] = 'HEROIN'
        elif 'winehouse' in filepath:
            dict['drug'] = 'ALCOHOL'
        elif 'barrett' in filepath:
            dict['drug'] = 'LSD'
        else:
            pass
        ADDICT_FACES_DICT.append(dict)
