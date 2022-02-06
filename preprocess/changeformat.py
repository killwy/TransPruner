import pandas
train=pandas.read_csv("Sceneflow_train.csv")
test=pandas.read_csv("Sceneflow_valid.csv")
left_list=[]
right_list=[]
disp_list=[]
for i in range(len(test)):
    left_list.append(test.iloc[i,1].replace("/data","/home/FastDataLoader"))
    right_list.append(test.iloc[i,2].replace("/data","/home/FastDataLoader"))
    disp_list.append(test.iloc[i,3].replace("/data","/home/FastDataLoader"))
df1=pandas.DataFrame({'left':left_list,'right':right_list,'disp':disp_list})
df1.to_csv('Sceneflow_valid1.csv')
left_list=[]
right_list=[]
disp_list=[]
for i in range(len(train)):
    left_list.append(train.iloc[i,1].replace("/data","/home/FastDataLoader"))
    right_list.append(train.iloc[i,2].replace("/data","/home/FastDataLoader"))
    disp_list.append(train.iloc[i,3].replace("/data","/home/FastDataLoader"))
df2=pandas.DataFrame({'left':left_list,'right':right_list,'disp':disp_list})
df2.to_csv('Sceneflow_train1.csv')