import os
import sys

output_list = ['Hips', 'LeftFoot', 'RightFoot']
input_list = ['Head', 'LeftHand', 'RightHand']
output_xyz = []
input_xyz = []
for i in output_list:
    output_xyz.append(i + ".X")
    output_xyz.append(i + ".Y")
    output_xyz.append(i + ".Z")
for i in input_list:
    input_xyz.append(i + ".X")
    input_xyz.append(i + ".Y")
    input_xyz.append(i + ".Z")

path = "./data/1_18/"
filename = "run_Take_2019-01-18_05.11.33_PM"
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
rotations_df.rename(columns=lambda x: str(x).split("_")[-1], inplace=True)
worldpos_df.rename(columns=lambda x: str(x).split("_")[-1], inplace=True)

df_time = worldpos_df["Time"]
df_worldpos_input = worldpos_df[input_xyz]
df_worldpos_output = worldpos_df[output_xyz]
df_rotations_input = rotations_df[input_xyz]
df_rotations_output = rotations_df[output_xyz]
df_worldpos_input.rename(columns=lambda x: str(x) + "_worldpos", inplace=True)
df_worldpos_output.rename(columns=lambda x: str(x) + "_worldpos", inplace=True)
df_rotations_input.rename(columns=lambda x: str(x) + "_rotations", inplace=True)
df_rotations_output.rename(columns=lambda x: str(x) + "_rotations", inplace=True)
##100 나누기
# df_worldpos_input /= 100
# df_worldpos_output /= 100
# df_rotations_input /= 100
# df_rotations_output /= 100

print(df_worldpos_input)
out_df = pandas.concat([df_time, df_worldpos_input, df_rotations_input, df_worldpos_output, df_rotations_output],
                       axis=1)  # column bind
out_file_name = "preprocesiing_" + filename+"csv"
out_df.to_csv(out_file_name, sep=',', encoding='utf-8', index=False)
