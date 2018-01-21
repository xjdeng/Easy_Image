import dlib, os

def run():
    mypath = os.path.abspath(__file__)
    dir_path = os.path.dirname(mypath)
    return dlib.shape_predictor(dir_path + "/shape_predictor_68_face_landmarks.dat")