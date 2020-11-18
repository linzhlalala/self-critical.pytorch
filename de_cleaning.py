import matplotlib.pyplot as plt
from pydicom import dcmread
import pandas as pd
import numpy as np
import os
import csv

remove_tags = [
    'PatientBirthDate',
    'PatientID',
    'PatientName',
    'PatientSex', ]

save_info = {
    # 'ImplementationVersionName':    [0x0002,0x0013], #'OFFIS_DCMTK_360'
    'ImageType': [0x0008, 0x0008],  # ['ORIGINAL', 'PRIMARY', 'COLOR']
    'SOPClassUID': [0x0008, 0x0016],  # VL Photographic Image Storage
    'StudyTime': [0x0008, 0x0030],  # 093540
    'ContentTime': [0x0008, 0x0033],  # 'Proofsheet'
    'Modality': [0x0008, 0x0060],  # 'OP' or 'XC'?
    'ConversionType': [0x0008, 0x0064],  # 'SI'
    'Manufacturer': [0x0008, 0x0070],  # Zeiss
    'StationName': [0x0008, 0x1010],  # VISU-CAPTURE1
    'SeriesDescription': [0x0008, 0x103e],  # 'Single FA 5:14.10 55¡ã ART'
    'PatientOrientation': [0x0020, 0x0020],  #
    'Laterality': [0x0020, 0x0060],  # L or R?
    'ImageLaterality': [0x0020, 0x0062],  # L or R?
    'PhotometricInterpretation': [0x0028, 0x0004],  # RGB?
    'ManufacturersModelName': [0x0008, 0x1090],  # "FF450 Plus"
}


def dcm_deid(ds):
    for tag in remove_tags:
        if tag in ds:
            ds[tag].value = ''
    ds.PatientIdentityRemoved = 'YES'


def dcm_info(ds):
    rt = {}
    for k, v in save_info.items():
        if v in ds:
            rt[k] = ds[v].repval
    return rt


def main(output_folder):
    freport = '/media/hdd/data/imcaption/FAsample/sample.csv'
    reportlist = pd.read_csv(freport, encoding='gb18030')

    table = []
    for index, row in reportlist.iterrows():
        id = row['id']
        print(id)
        finding = row['Findings']
        impression = row['Impression']

        folder = os.walk(id)
        # print(folder)
        for path, dir_list, file_list in folder:
            save_to_path = os.path.join(output_folder, path)
            if not os.path.exists(save_to_path):
                os.makedirs(save_to_path)
            for idx, dcm_file in enumerate(file_list):
                dcm_filepath = os.path.join(path, dcm_file)
                ds = dcmread(dcm_filepath)
                dcm_deid(ds)
                dcm_dict = dcm_info(ds)
                dcm_dict['studyid'] = id
                dcm_dict['imgid'] = dcm_file.replace('.dcm', '')
                dcm_dict['filepath'] = dcm_filepath
                table.append(dcm_dict)

                # save_to_path = os.path.join(output_folder,path)
                # ds.save_as(save_to_file)

        if index == 500:
            break

    csv_file = "infos.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        cols = list(save_info.keys())
        cols.extend(['studyid', 'imgid', 'filepath'])
        writer = csv.DictWriter(csvfile, fieldnames=cols)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    output_folder = "output"
    main(output_folder)
    print("Finished")