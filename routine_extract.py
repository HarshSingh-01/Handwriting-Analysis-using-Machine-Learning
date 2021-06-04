import os 
import extractor

train_path = "C:\\Users\Harsh\Desktop\Projects\Datasets\IAM Handwriting\Images"
test_path = "C:\\Users\Harsh\Desktop\Projects\Handwriting-Analysis-using-Machine-Learning\Test Images"
feature_list_path = "C:\\Users\Harsh\Desktop\Projects\Handwriting-Analysis-using-Machine-Learning\Extracted_Features/raw_feature_list.txt"

os.chdir(test_path)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
os.chdir('..')

page_ids = []
if os.path.isfile(feature_list_path):
    print("Info: raw feature list already exists..")
    with open(feature_list_path, 'r') as label:
        for line in label:
            content = line.split()
            page_id = content[-1]
            page_ids.append(page_id)


with open(feature_list_path,'a') as label:
    count = len(page_ids)
    for file_name in files:
        if file_name in page_ids:
            continue
        features = extractor.start(file_name)
        features.append(file_name) 
        for i in features:
            label.write("%s\t" % i)
            # label.write("%s\n"% i)
#             print(label, end='')
            count += 1
            progress = (count*100)/len(files)
            print(str(count)+' '+ file_name+' '+ str(progress)+'%')
        label.write("\n")
        print('done')
