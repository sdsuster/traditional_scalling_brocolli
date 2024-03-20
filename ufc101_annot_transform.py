import os
OUTPUT_FORMAT = './dataset/ucf101_{}_{}.csv'
INPUT_FORMAT = '/mnt/d/Datasets/UCF101_Annot/{}list{:02d}.txt'

UCF101_FOLDER = '/mnt/d/Datasets/UCF101/UCF-101/'
FLOW_FOLDER = '/mnt/d/Datasets/UCF101_Flow/'
headers=['file', 'flow_file', 'label']

### map class into hashtable
class_table = {}
with open('/mnt/d/Datasets/UCF101_Annot/classInd.txt', "r") as f:
    for l in f.readlines():
        index, label = l.replace('\n', '').split(' ')
        class_table[label] = int(index) - 1

    f.close()

for i in range(3):
    with open(INPUT_FORMAT.format('train',i+1), "r") as f:
        o = open(OUTPUT_FORMAT.format('train', i+1), "w")
        o.write(",".join(headers) + "\n")
        list = []
        for l in f.readlines():
            file_path, index = l.replace('\n', '').split(' ')
            list.append(f'{os.path.join(UCF101_FOLDER, file_path)},{os.path.join(FLOW_FOLDER, file_path)},{int(index) - 1}\n')

        o.writelines(list)
        o.close()
        f.close()

        #test
        
    with open(INPUT_FORMAT.format('test',i+1), "r") as f:
        o = open(OUTPUT_FORMAT.format('test', i+1), "w")
        o.write(",".join(headers) + "\n")
        list = []
        for l in f.readlines():
            file_path = l.replace('\n', '')
            index = class_table[file_path.split('/')[0]]
            list.append(f'{os.path.join(UCF101_FOLDER, file_path)},{os.path.join(FLOW_FOLDER, file_path)},{int(index)}\n')

        o.writelines(list)
        o.close()
        f.close()