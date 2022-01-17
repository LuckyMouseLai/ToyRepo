import dlib

recognizer = dlib.face_recognition_model_v1("./data/repo_dlib/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/repo_dlib/shape_predictor_68_face_landmarks.dat')