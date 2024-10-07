import csv
import os


data_dir = "DATA_FILE/test_train/train/clean_signals"
EXT = 'wav'
RETURN_FULLPATH = True


columns = ['path', 'filename','subset']
rows = []
i = 0


for root, _, files in os.walk(data_dir, topdown=False):
    for name in files:
        i=i+1
        if os.path.isfile(os.path.join(root, name)):
            if name.endswith(EXT):
                file_path = os.path.join(root, name)
                file_name = os.path.basename(name)
            if RETURN_FULLPATH:
                if i<10:
                    rows.append([os.path.join(root),name, 'validation'])
                


                rows.append([os.path.join(root),name, 'training'])
            else:
                rows.append([name, 'training'])

print(len(rows) )
with open('metrics_file.csv', 'w') as f:
#with open('DATA_FILE/test_train/train/meta2.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)
    write.writerows(rows)
