import numpy as np
import pandas as pd


def data_process(train_Data):
    if "income" in train_Data.columns:
        Data = train_Data.drop(["sex", "income"], axis=1)
    else:
        Data = train_Data.drop(["sex"], axis=1)
    # 读取非数字的列
    listNoNumColumn = [col for col in Data.columns if Data[col].dtype == "object"]
    #  读取数字的列
    listNumColumn = [x for x in list(Data) if x not in listNoNumColumn]
    NoNumColumn = Data[listNoNumColumn]
    NumColumn = Data[listNumColumn]
    #  性别女为1，男0
    NumColumn.insert(0, "sex", (train_Data["sex"] == " Female").astype(np.int))
    NoNumColumn = pd.get_dummies(NoNumColumn)
    Data = pd.concat([NumColumn, NoNumColumn], axis=1)
    Data_x = Data.astype("int64")
    # 归一化
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()
    Data_x = Data_x.values
    return Data_x


def process_label(train_Data):
    label = train_Data["income"]
    Data_y = pd.DataFrame((label == ' >50K').astype("int64"), columns=["income"])
    return Data_y


train_Data = pd.read_csv("./data/train.csv")
test_Data = pd.read_csv("./data/test.csv")
trainX = data_process(train_Data)
testX = data_process(test_Data)
trainy = process_label(train_Data)
print(trainy)
