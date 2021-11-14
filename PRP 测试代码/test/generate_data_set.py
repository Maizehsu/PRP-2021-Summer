import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import argparse
from osgeo import osr

parser = argparse.ArgumentParser(
    description="《交通大数据：理论与方法》样例数据生成。")
parser.add_argument('-d', '--data', nargs='?', default=os.getcwd(),
                    type=str, help="数据存储路径。")
parser.add_argument('-v', '--version', action='version', 
                    version='1.0', help="版本信息。")
_args = parser.parse_args()

data_path = Path(_args.data)