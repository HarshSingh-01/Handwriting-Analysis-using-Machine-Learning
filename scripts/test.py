import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from PythonScripts import extract
from scripts import extractor
from scripts import categorize
# from pathlib import Path

model_folder_path = "models/"
test_path = "Test Images/"


def test(classifier):
    if bool(classifier):
        a = "Present"
    else:
        a = "Absent"
    return a
    

def predict(file_name, kernel="rbf"):
    prediction_list = []
    features_dict = {}
    personality_dict = {}

    clf1 = model_folder_path + kernel + "/clf1.sav"
    # clf2 = model_folder_path + kernel + "/clf2.sav"
    clf3 = model_folder_path + kernel + "/clf3.sav"
    clf4 = model_folder_path + kernel + "/clf4.sav"
    clf5 = model_folder_path + kernel + "/clf5.sav"
    clf6 = model_folder_path + kernel + "/clf6.sav"
    clf7 = model_folder_path + kernel + "/clf7.sav"
    clf8 = model_folder_path + kernel + "/clf8.sav"

    clf1 = pickle.load(open(clf1, 'rb'))
    # clf2 = pickle.load(open(clf2, 'rb'))
    clf3 = pickle.load(open(clf3, 'rb'))
    clf4 = pickle.load(open(clf4, 'rb'))
    clf5 = pickle.load(open(clf5, 'rb'))
    clf6 = pickle.load(open(clf6, 'rb'))
    clf7 = pickle.load(open(clf7, 'rb'))
    clf8 = pickle.load(open(clf8, 'rb'))    

    raw_features = extractor.start(file_name)

    raw_baseline_angle = raw_features[0]
    baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
    features_dict["Baseline Angle"] = comment

    raw_top_margin = raw_features[1]
    top_margin, comment = categorize.determine_top_margin(raw_top_margin)
    features_dict["Top Margin"] = comment

    raw_letter_size = raw_features[2]
    letter_size, comment = categorize.determine_letter_size(raw_letter_size)
    features_dict["Letter Size"] = comment

    raw_line_spacing = raw_features[3]
    line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
    features_dict["Line Spacing"] = comment

    raw_word_spacing = raw_features[4]
    word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
    features_dict["Word Spacing"] = comment

    # raw_pen_pressure = raw_features[5]
    # pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
    # features_dict["Pen Pressure: "] = comment

    raw_slant_angle = raw_features[5]
    slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
    features_dict["Slant Angle"] = comment

    # Personality traits
    p1 = test(clf1.predict([[baseline_angle, slant_angle]])[0])
    # p2 = test(clf2.predict([[letter_size, pen_pressure]])[0])
    p3 = test(clf3.predict([[letter_size, top_margin]])[0])
    p4 = test(clf4.predict([[line_spacing, word_spacing]])[0])
    p5 = test(clf5.predict([[slant_angle, top_margin]])[0])
    p6 = test(clf6.predict([[letter_size, line_spacing]])[0])
    p7 = test(clf7.predict([[letter_size, word_spacing]])[0])
    p8 = test(clf8.predict([[line_spacing, word_spacing]])[0])

    personality_dict["Emotional Stability"] = p1

    # personality_dict["Mental Energy or Will Power"] = p2

    personality_dict["Modesty"] = p3
    
    personality_dict["Personal Harmony and Flexibility"] = p4

    personality_dict["Lack of Discipline"] = p5

    personality_dict["Poor Concentration"] = p6

    personality_dict["Non Communicativeness"] = p7

    personality_dict["Social Isolation"] = p8

    print("\nHandwritting Features\n")
    for k, v in features_dict.items():
        print(f"{k}: {v}")

    print("\nPersonality Traits\n")
    for k, v in personality_dict.items():
        print(f"{k}: {v}")

    return {"features":features_dict, "traits":personality_dict}

    


def output(Image, kernel):
    path = test_path + Image
    predict(path, kernel)
    # img = mpimg.imread(path)
    # imgplot = plt.imshow(img)
    plt.show()

# output("Img (2).jpg", kernel="rbf") # rbf or poly