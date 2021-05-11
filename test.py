import cv2
import numpy as np
from scripts import extract
from scripts import categorize


def testFunction(filename):
    raw_feature = extract.start(filename)
    print(raw_feature)

testFunction("a01-000u.png")