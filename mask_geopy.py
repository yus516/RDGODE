import numpy as np
import pandas as pd
import math
import geopy.distance


df = pd.read_csv('graph_sensor_locations.csv')
# print(df)

mask1 = np.load('mask_10[0-4].npy')
mask2 = np.load('mask_10[4-8].npy')
mask3 = np.load('mask_10[8-12].npy')
mask4 = np.load('mask_10[12-16].npy')
mask5 = np.load('mask_10[16-20].npy')
mask6 = np.load('mask_10[20-24].npy')

def process_mask(mask, df):
    _10th_col = mask[:, 10]
    _10th_row = mask[10, :]
    _10th_col_top3_indices = np.argsort(_10th_col)[::-1][:3]
    _10th_row_top3_indices = np.argsort(_10th_row)[::-1][:3]
    print(_10th_col_top3_indices)
    print(_10th_row_top3_indices)
    coord_df_10 = (df.iloc[10]['latitude'], df.iloc[10]['longitude'])
    # df_10_lat = df.iloc[10]['latitude']
    # df_10_long = df.iloc[10]['longitude']
    col_dis = 0
    row_dis = 0
    for c in _10th_col_top3_indices:
        coord_c = (df.iloc[c]['latitude'], df.iloc[c]['longitude'])
        col_dis = col_dis + geopy.distance.geodesic(coord_df_10, coord_c).km
        # col_dis = col_dis + math.sqrt((df.iloc[c]['latitude'] - df_10_lat)**2 + (df.iloc[c]['longitude'] - df_10_long)**2)
    for r in _10th_row_top3_indices:
        coord_r = (df.iloc[r]['latitude'], df.iloc[r]['longitude'])
        row_dis = row_dis + geopy.distance.geodesic(coord_df_10, coord_r).km
        # row_dis = row_dis + math.sqrt((df.iloc[r]['latitude'] - df_10_lat)**2 + (df.iloc[r]['longitude'] - df_10_long)**2)
    col_dis_avg = col_dis / 3
    row_dis_avg = row_dis / 3
    return col_dis_avg, row_dis_avg

col_dis_avg1, row_dis_avg1 = process_mask(mask1, df)
col_dis_avg2, row_dis_avg2 = process_mask(mask2, df)
col_dis_avg3, row_dis_avg3 = process_mask(mask3, df)
col_dis_avg4, row_dis_avg4 = process_mask(mask4, df)
col_dis_avg5, row_dis_avg5 = process_mask(mask5, df)
col_dis_avg6, row_dis_avg6 = process_mask(mask6, df)


print("col_dis_avg1: ", col_dis_avg1, "row_dis_avg1: ", row_dis_avg1)
print("col_dis_avg2: ", col_dis_avg2, "row_dis_avg2: ", row_dis_avg2)
print("col_dis_avg3: ", col_dis_avg3, "row_dis_avg3: ", row_dis_avg3)
print("col_dis_avg4: ", col_dis_avg4, "row_dis_avg4: ", row_dis_avg4)
print("col_dis_avg5: ", col_dis_avg5, "row_dis_avg5: ", row_dis_avg5)
print("col_dis_avg6: ", col_dis_avg6, "row_dis_avg6: ", row_dis_avg6)

