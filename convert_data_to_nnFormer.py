import dicom2nifti
import os
import sys
import tempfile
import tqdm
import shutil
import nibabel as nib
import json
import numpy as np

from pathlib import Path


# class_map_chestCT = {
#     1: "Bronchial",
#     2: "PulmonaryVessels",
# } 

# ## Bronchial
# class_map_chestCT = {
#     1: "Bronchial"
# }

# ## TotalPulmonaryVessels
# class_map_chestCT = {
#     1: "TotalPulmonaryVessels"
# }

class_map_chestCT = {
    "1": "Skin",
    "2": "Lung",
    "3": "Bone",
    "4": "Heart",
    "5": "PulmonaryVessels",
    "6": "Bronchial"
}


# class_map_chestCT = {
#     1: "Sphere",
# }

def combine_labels(ref_img, file_out, masks):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)  
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


def generate_json_from_dir_v2(nnunet_path, subjects_train, subjects_val, subject_test, labels, name):
    print("Creating dataset.json...")
    # out_base = Path(os.environ['nnUNet_raw']) / foldername
    out_base = nnunet_path
    foldername = out_base.name
    
    json_dict = {}
    json_dict['description'] = "Segmentation of nnFormer classes"
    json_dict['reference'] = "https://zenodo.org/record/6802614"
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "2.0"
    json_dict['modality'] = {"0": "CT"}
    # json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    labels.update({"0": "background"})
    json_dict['labels'] = {"0": "background"}
    json_dict['labels'].update(labels)
    json_dict['name'] = name
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict["numTest"] = len(subject_test)
    json_dict["tensorImageSize"] = "4D"
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'
    json_dict["training"] = []
    json_dict["test"] = []

    for i in subject_test:
        json_dict["training"].append(f"./imagesTs/{j}.nii.gz")
        
    for j in (subjects_train + subjects_val):
        tmp = {
            "image": f"./imagesTr/{j}.nii.gz",
            "label": f"./labeslTr/{j}.nii.gz"
        }
        json_dict["training"].append(tmp)

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnFormer_preprocessed']) / foldername
    output_folder_pkl.mkdir(exist_ok=True)
    print(output_folder_pkl)

    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


if __name__ == "__main__":
    # dataset_path = Path(sys.argv[1])  # directory containining all the subjects
    # nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset
    # TotalSegmentator is made up of 5 models. Choose which one you want to produce. Choose from: 
    #   class_map_part_organs
    #   class_map_part_vertebrae 
    #   class_map_part_cardiac 
    #   class_map_part_muscles 
    #   class_map_part_ribs
    # class_map_name = sys.argv[3]  

    # dataset_path = "/mnt/Data/data03/dst/"
    # dataset_path = Path("/data/dulicui/project/git/nnUNet/dataset/ori_nii/data00/")
    # nnunet_path = Path("/data/dulicui/project/git/nnUNet/dataset/nnUnet_raw/")
    # dataset_name = "Data555_LungSeg"
    
 
    dataset_path = Path("/mnt/SEData/data00_nii/")
    # dataset_path = Path("/mnt/SEData/data03_nii/")

    # # nnunet_path = Path("/mnt/Data01/nnFormer/dataset/nnFormer_raw/nnFormer_raw_data/")
    # dataset_name = "Task010_data00_cls6_test"

    nnunet_path = Path("/mnt/Data02/project/nnFormer/dataset/nnFormer_raw/nnFormer_raw_data/")
    dataset_name = "Task559_data00_cls6"

    # train_uid_file = "/media/ubuntu/Elements SE/data00/dst/500_100/train.txt"
    # valid_valid_file = "/media/ubuntu/Elements SE/data00/dst/500_100/valid.txt"

    # train_uid_file = "/media/ubuntu/Elements SE/data13/dst/380_133/train.txt"
    # valid_valid_file = "/media/ubuntu/Elements SE/data13/dst/380_133/valid.txt"

    train_uid_file = "/mnt/SEData/data00/dst/500_100/train.txt"
    valid_valid_file = "/mnt/SEData/data00/dst/500_100/valid.txt"

    # train_uid_file = "/mnt/SEData/data03/dst_combine/490_188/train.txt"
    # valid_valid_file = "/mnt/SEData/data03/dst_combine/490_188/valid.txt"

    nnunet_path = nnunet_path / dataset_name

    class_map = class_map_chestCT

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)


    # meta = pd.read_csv(dataset_path / "meta.csv", sep=";")
    # subjects_train = samples[:8]
    # subjects_val = samples[8:9]
    # subjects_test = samples[9:10]

    samples = os.listdir(dataset_path / "image")

    def read_txt(uid_file):
        with open(uid_file) as f:
            uids = [x.strip() for x in f.readlines()]
        return uids
    subjects_train = read_txt(train_uid_file)
    subjects_val = read_txt(valid_valid_file)
    subjects_test = []

    # import pdb; pdb.set_trace()
    for uid in ["20191209-100016", "20191220-100016"]:
        if uid in subjects_train:
            subjects_train.remove(uid)
        elif uid in subjects_val:
            subjects_val.remove(uid)
        elif uid in subjects_test:
            subjects_test.remove(uid)
        else:
            print("Error: No such files in dataset!")

    print(f"train: {len(subjects_train)}, val: {len(subjects_val)}, test: {len(subjects_test)}")
    # #####===== debug=======
    # subjects_train = subjects_train[:5]
    # subjects_val = subjects_val[:1]
    # subjects_test = subjects_test[:1]
    # #####===== debug=======

    # #####  ##TotalPunmonaryVessels 有问题的数据重新生成debug============
    # ss = [
    #         "20191216-100065", "20210604-100023","20211228-100005","20211229-100017","20211230-100027",
    #         "20220104-100011","20220104-100047","20220105-100024","20220105-100053","20220106-100026"
    #     ]

    # subjects_train = [i for i in subjects_train if i in ss]
    # subjects_val = [i for i in subjects_val if i in ss]

    # print(f"train: {len(subjects_train)}, val: {len(subjects_val)}, test: {len(subjects_test)}")
    # #####debug============
    

    # import pdb; pdb.set_trace()
    # cnt = 0
    # print("Copying train data...")
    # for subject in (subjects_train + subjects_val): #[440:]
    #     if subject == "20191209-100016": ## data00
    #             continue
    #     if subject == "20191220-100016": ## data13
    #         continue
    #     if subject == "20190725-000030": ## data02
    #             continue
    #     cnt += 1
    #     # print(f"dealing {cnt}: {subject}")
        
    #     # if subject not in [
    #     #     "20191216-100065", "20210604-100023","20211228-100005","20211229-100017","20211203-100027",
    #     #     "20220104-100011","20220104-100047","20220105-100024","20220105-100053","20220106-100026"
    #     # ]:
    #     #     continue

    #     print(f" IN dealing: {subject}")
    #     subject_path = dataset_path / "image" / f"{subject}.nii.gz"
    #     subject_name = subject.split(".")[0]
    #     shutil.copy(subject_path, nnunet_path / "imagesTr" / f"{subject_name}_0000.nii.gz")  ## 执行完之后进行软链，image拷贝一份就可以
    #     combine_labels(subject_path,
    #                    nnunet_path / "labelsTr" / f"{subject_name}.nii.gz",
    #                 #    [dataset_path / f"{roi}" / f"{roi}_{subject_name}.nii.gz" for roi in class_map.values()]
    #                    [dataset_path / "mask" / f"{roi}" / f"{roi}_{subject_name}.nii.gz" for roi in class_map.values()]
    #     )

    # print("Copying test data...")
    # # for subject in tqdm(subjects_test):
    # for subject in (subjects_test):
    #     subject_path = dataset_path / "image" / subject
    #     subject_name = subject.split(".")[0]
    #     shutil.copy(subject_path, nnunet_path / "imagesTs" / f"{subject_name}_0000.nii.gz")
    #     combine_labels(subject_path,
    #                    nnunet_path / "labelsTs" / f"{subject_name}.nii.gz",
    #                 #    [dataset_path / f"{roi}" / f"{roi}_{subject_name}.nii.gz" for roi in class_map.values()]
    #                    [dataset_path / "mask" / f"{roi}" / f"{roi}_{subject_name}.nii.gz" for roi in class_map.values()]
    #     ) 


    name = "_".join(dataset_name.split("_")[1:])
    generate_json_from_dir_v2(nnunet_path, subjects_train, subjects_val, subjects_test, class_map, name)   
        

