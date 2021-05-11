def determine_baseline_angle(raw_baseline_angle):
    comment = ""
    # Falling
    if(raw_baseline_angle >= 0.2):
        baseline_angle = 0
        comment = "DESCENDING"

    # Rising
    elif(raw_baseline_angle <= -0.3):
        baseline_angle = 1
        comment = "ASCENDING"

    # Straight
    else: 
        baseline_angle = 2
        comment = "STRAIGHT"
    
    return baseline_angle, comment

def determine_top_margin(raw_top_margin):
    comment = ""
    # medium and bigger
    if(raw_top_margin >= 1.7):
        top_margin = 0
        comment = "MEDIUM OR BIGGER"

    # Narrow
    else: 
        top_margin = 1
        comment = "NARROW"

    return top_margin, comment

def determine_letter_size(raw_letter_size):
    comment = ""
    # Big
    if(raw_letter_size >= 18.0):
        letter_size = 0
        comment = "BIG"

    # Small
    elif(raw_letter_size < 13.0):
        letter_size = 1
        comment = "SMALL"

    # Medium
    else:
        letter_size = 2
        comment = "MEDIUM"

    return letter_size, comment

def determine_line_spacing(raw_line_spacing):
    comment = ""
    # Big
    if(raw_line_spacing >= 3.5):
        line_spacing = 0
        comment = "BIG"
    
    # Small
    elif(raw_line_spacing < 2.0):
        line_spacing = 1
        comment = "SMALL"

    # Medium
    else:
        line_spacing = 2
        comment = "MEDIUM"

    return line_spacing, comment

def determine_word_spacing(raw_word_spacing):
    comment = ""
    # Big
    if(raw_word_spacing > 2.0):
        word_spacing = 0
        comment = "BIG"

    # Small
    elif(raw_word_spacing < 1.2):
        word_spacing = 1
        comment = "SMALL"

    # Medium
    else:
        word_spacing = 2
        comment = "MEDIUM"

    return word_spacing, comment

def determine_pen_pressure(raw_pen_pressure):
    comment = ""
    # Heavy
    if(raw_pen_pressure > 180.0):
        pen_pressure = 0
        comment = "HEAVY"

    # Light
    elif(raw_pen_pressure < 151.0):
        pen_pressure = 1
        comment = "LIGHT"

    # Medium
    else:
        pen_pressure = 2
        comment = "MEDIUM"

    return pen_pressure, comment

def determine_slant_angle(raw_slant_angle):
    comment = ""
    # Extremely Reclined
    if(raw_slant_angle == -45.0 or raw_slant_angle == -30.0):
        slant_angle = 0
        comment = "EXTREMELY RECLINED"
    
    # A little reclined or moderately reclined
    elif(raw_slant_angle == -15.0 or raw_slant_angle == -5.0):
        slant_angle = 1
        comment = "A LITTLE OR MODERATELY RECLINED"
    
    # A little inclined
    elif(raw_slant_angle == 5.0 or raw_slant_angle == 15.0):
        slant_angle = 2
        comment = "A LITTLE INCLINED"

    # Moderately inclined
    elif(raw_slant_angle == 30.0):
        slant_angle  = 3
        comment = "MODERATELY INCLINED"

    # Extremely inclined
    elif(raw_slant_angle == 0.0):
        slant_angle = 4
        comment = "EXTREMELY INCLINED"

    # Straight
    elif(raw_slant_angle == 0.0):
        slant_angle = 5
        comment = "STRAIGHT"

    #IRREGULAR
    else:
        slant_angle = 6
        comment = "IRREGULAR"

    return slant_angle, comment