import os
import sys

path = "./data/1_3/"
filename = "test_Take_2019-01-03_03.48.29_PM"
filepath = os.path.join(path, filename + ".bvh")
worldpos_file_name = filename + "_worldpos.csv"
rotations_file_name = filename + "_rotations.csv"
worldpos_file_path = os.path.join(path, worldpos_file_name)
if not os.path.isfile(worldpos_file_path):
    os.system("bvh-converter " + filepath)  # csv 생성
rotations_file_path = os.path.join(path, rotations_file_name)
if not os.path.isfile(rotations_file_path):
    os.system("bvh-converter -r " + filepath)
print("Hello")
import pandas

# read csv file
rotations_df = pandas.read_csv(rotations_file_path)  # df is pandas.DataFrame
worldpos_df = pandas.read_csv(worldpos_file_path)  # df is pandas.DataFrame
df1 = worldpos_df[['Skeleton6_Hips.X', 'Skeleton6_Hips.Y']]
rotations_df.rename(columns=lambda x: str(x).split("_")[-1], inplace=True)
worldpos_df.rename(columns=lambda x: str(x).split("_")[-1], inplace=True)
columns_list = worldpos_df.columns

print("##### data #####")
index = 0
for i in rotations_df.columns.T.values:
    if index % 3 == 1:
        print(str(i).split(".")[0].split("_")[-1])
    index += 1
