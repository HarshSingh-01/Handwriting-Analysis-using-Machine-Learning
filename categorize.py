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




