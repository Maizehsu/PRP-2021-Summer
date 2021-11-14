# basic packages
import pandas as pd
import numpy as np
import time
import os
from utm import *
from tqdm import tqdm, tqdm_pandas
from osgeo import osr
import ref_trans

# 预设地址和其他全局变量
feature_file_name = '1102 8am-8pm 21-11-11'
raw_data_path = 'F:/大学/第40期PRP/交通订单数据/traffic_data/gps_20161102.csv'
feature_dst_path = 'F:/大学/第40期PRP/特征提取/1_feature_analysis/' + feature_file_name + '.csv'
date = '20161102'
#在此处设置时间窗(单位为3秒)和空间网格的边长(WGS84坐标系)
time_interval = 100
space_interval = 70
# 滞留时间阈值，超过阈值视为无效订单


# 设置时间区间 读取原数据
# 时间区间: 减少单次的处理量
time1 = '2016 11 02 08:00:00'
time2 = '2016 11 02 20:00:00'
stamp1 = time.mktime(time.strptime(time1, '%Y %m %d %H:%M:%S'))
stamp2 = time.mktime(time.strptime(time2, '%Y %m %d %H:%M:%S'))

#导入原地理数据
df = pd.read_csv(raw_data_path, header = None) #注意我此处使用的是移动硬盘的地址
df.columns = ['driver_ID', 'order_ID', 'timestamp', 'lon', 'lat']
df.timestamp = df.timestamp + 8*3600
## 只取预设时间区间内的数据
df = df[(df['timestamp'] >= stamp1)&(df['timestamp'] < stamp2)].reset_index(drop = True)

# 将空间坐标转换为WGS-84(耗时会很长)
xy = df[['lon','lat']].apply(lambda x: ref_trans.gcj02_to_wgs84(x[0],x[1])[:2], axis = 1)
df['lon'] = [x[0] for x in xy]
df['lat'] = [x[1] for x in xy]

# 再把WGS-84转换为UTM平面直角系(保留WGS-84数据)
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
# 2.Pseudo-Mercator
inp = osr.SpatialReference()
inp.ImportFromEPSG(3857)
# 3.定义坐标变换映射
transformation = osr.CoordinateTransformation(wgs84, inp)
# 4.转换原数据的坐标
xy = df[['lon','lat']].apply(lambda x: transformation.TransformPoint(x[0],x[1])[:2], axis = 1)
# 5.写入df
df['x'] = [x[0] for x in xy]
df['y'] = [x[1] for x in xy]

# 时间窗划分
df['time_ID'] = df.timestamp.apply(lambda x: (x - stamp1)//time_interval)

# 空间网格划分
# 1.计算左边界和上边界，左右-x， 上下-y
left = df['x'].min()
up = df['y'].max()

# 2.生成横向和纵向索引
df['row_id'] = df['y'].apply(lambda y: (up - y)//space_interval)
df['col_id'] = df['x'].apply(lambda x: (x - left)//space_interval)

df = df.dropna()

# 下面开始时空特征提取

#1. 计算瞬时速度

# 排序：先按司机排，同司机按订单排，同订单再按时间排
df = df.sort_values(by = ['driver_ID', 'order_ID', 'timestamp']).reset_index(drop = True)

# 将订单id下移一行，用于判断前后数据是否属于同一订单
df['orderFlag'] = df['order_ID'].shift(1)
df['identi'] = (df['orderFlag'] == df['order_ID']) #一个由boolean构成的列，方便后面所有shift完成了之后再删除分界行

# 将坐标，时间戳下移一行，匹配相应轨迹点
df['x1'] = df['x'].shift(1)
df['y1'] = df['y'].shift(1)
df['timestamp1'] = df['timestamp'].shift(1)

# 将不属于同一订单的轨迹点删除
df = df[df['identi'] == True]

# 计算相邻轨迹点之间的距离和相差时间
# 距离采用欧式距离
dist = np.sqrt(np.square(df['x'].values - df['x1'].values) + np.square(df['y'].values - df['y1'].values))
time = df['timestamp'].values - df['timestamp1'].values

# 计算速度
df['speed'] = dist/time

# 删除临时数据
df = df.drop(columns = ['x1', 'y1', 'orderFlag', 'timestamp1', 'identi'])

# 2.计算瞬时加速度
df['speed1'] = df['speed'].shift(1)
df['timestamp1'] = df['timestamp'].shift(1)
df['identi'] = df['order_ID'].shift(1)

df = df[df.identi == df.order_ID]

df['acc'] = (df.speed - df.speed1)/(df.timestamp - df.timestamp1)

df = df.drop(columns = ['speed1', 'timestamp1', 'identi'])
df = df.reset_index(drop = True)

# 下面计算集体/网格平均特征

# 1. 网格平均速度：先求每辆车在网格中的平均速度，然后求网格中所有个体平均速度的军制
# 基于时空网格和估计id分组
orderGrouped = df.groupby(['row_id', 'col_id', 'time_ID', 'order_ID'])
# 网格在每个时刻（时间窗）的平均速度
grouped_speed = orderGrouped.speed.mean().reset_index()
grouped_speed = grouped_speed.groupby(['row_id', 'col_id', 'time_ID'])
grid_speed = grouped_speed.speed.mean()
# 去除异常值
grid_speed = grid_speed.clip(grid_speed.quantile(0.05), grid_speed.quantile(0.95))

# 2. 网格平均加速度
gridGrouped = df.groupby(['row_id', 'col_id', 'time_ID'])
grid_acc = gridGrouped.acc.mean()

# 3.网格浮动车流量
grouped_volume = orderGrouped.speed.last().reset_index() #每个时空网格中的每个order只保留一辆（用last（）来取）
grouped_volume = grouped_volume.groupby(['row_id', 'col_id', 'time_ID'])
grid_volume = grouped_volume['speed'].size()
grid_volume = grid_volume.clip(grid_volume.quantile(0.05), grid_volume.quantile(0.95))

# 4.网格车速标准差
grid_v_std = gridGrouped.speed.std(ddof=0)
# 去除异常值
grid_v_std = grid_v_std.clip(grid_v_std.quantile(0.05), grid_v_std.quantile(0.95))

# 5.网格平均停车次数
stopNum = gridGrouped.speed.agg(lambda x: (x==0).sum())
grid_stop = pd.concat((stopNum, grid_volume), axis = 1)
grid_stop['stopNum'] = stopNum.values/ grid_volume.values
grid_stop = grid_stop['stopNum']
grid_stop = grid_stop.clip(0, grid_stop.quantile(0.95))

# 下面进行数据整理
feature = pd.concat([grid_speed, grid_acc, grid_volume, grid_v_std, grid_stop], axis = 1).reset_index()
feature.columns = ['row_id','col_id', 'time_id', 'aveSpeed', 'gridAcc', 'volume', 'speedStd', 'stopNum']

feature.sort_values(['stopNum']).reset_index(drop=True)
feature['date'] = date

feature.to_csv(feature_dst_path, index = None)