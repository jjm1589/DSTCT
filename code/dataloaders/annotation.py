import os
import random

trainval_percent    = 0.8
train_percent       = 0.875

fhps_path = '/root/autodl-tmp/SSL4MIS/data/FHPS'
# Randomly assigned data
if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(fhps_path, 'data/slices')

    num = 5101
    list = range(num)
    tall = int(num * trainval_percent)
    tr = int(tall * train_percent)
    # tv = int(tall * 0.125)
    # tt = int(tall * 0.2)
    trainval = random.sample(list, tall)
    train = random.sample(trainval, tr)
    # val = random.sample(trainval, tv)
    # test = random.sample(trainval, tt)
    # print(len(trainval))
    # print(len(train))
    # print(len(val))
    # print(len(test))
    count = 0
    temp_seg = os.listdir(segfilepath)
    all_slices_list   = open(os.path.join(fhps_path, 'all_slices.list'), 'w')
    train_slices_list = open(os.path.join(fhps_path, 'train_slices.list'), 'w')
    val_list          = open(os.path.join(fhps_path, 'val.list'), 'w')
    test_list         = open(os.path.join(fhps_path, 'test.list'), 'w')

    total_seg = []
    for seg in temp_seg:
        total_seg.append(seg)
    
    for i in list:
        # print(total_seg[i])
        if i in trainval:
            # for j in range(i * 3, i * 3 + 3):
            #     all_slices_list.write(total_seg[j].split(".")[0] + '\n')
            all_slices_list.write(total_seg[i].split(".")[0] + '\n')
            if i in train:
                # print("train")
                # print(i)
                # for j in range(i * 3, i * 3 + 3):
                #     train_slices_list.write(total_seg[j].split(".")[0] + '\n')
                 train_slices_list.write(total_seg[i].split(".")[0] + '\n')
            else:
                # print("val")
                # print(i)
                count += 1
                fname = total_seg[i].split(".")[0]
                fname = fname.split("_")[0]
                val_list.write(fname + '\n')
                # for j in range(i * 3, i * 3 + 3):
                #     fname = total_seg[j].split(".")[0]
                #     fname = fname.split("_")[0]
                #     val_list.write(fname + '\n')
            # else:
            #     fname = total_seg[i].split(".")[0]
            #     fname = fname.split("_")[0]
            #     test_list.write(fname + '\n')
            #     # for j in range(i * 3, i * 3 + 3):
            #     #     fname = total_seg[j].split(".")[0]
            #     #     fname = fname.split("_")[0]
            #     #     test_list.write(total_seg[j].split(".")[0] + '\n')
        else :
            fname = total_seg[i].split(".")[0]
            fname = fname.split("_")[0]
            test_list.write(fname + '\n')
            

    print(count)
    print("Generate txt in ImageSets done.")
    all_slices_list.close()
    train_slices_list.close()
    val_list.close()
    test_list.close()
