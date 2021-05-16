import cv2
import numpy as np
# from PythonScripts import extract
# from scripts import categorize
import extractor


def testFunction(filename):
    raw_feature = extractor.start(filename)
    print(raw_feature)

testFunction("a01-000u.png")