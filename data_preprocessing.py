import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

'''
Index(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'],
      dtype='object')
'''

#导入数据
def load_data():
    train_df = pd.read_csv('train.csv',index_col=0)
    test_df = pd.read_csv('test.csv',index_col=0)
    return train_df, test_df

#合并训练集和测试集（必要）
#使训练集和测试集数据在同一尺度下，保证预测的准确性
def concat_data():
    train_df, test_df = load_data()
    train_df.pop('SalePrice')
    df = pd.concat((train_df,test_df))
    return df


#查看过滤后每一个特征的分布
#离散的用线箱图展示，连续的用散点图展示
def show_feature(train_df):
    train_df_filled = drop_na_fiture(train_df)
    train_df_filled = fill_nan_feature(train_df_filled)
    output = 'SalePrice'
    vars = train_df_filled.columns[1:]
    vars = vars[:-1]
    continuity_dict = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd",
                       # "MasVnrArea",
                       "BsmtFinSF1", "BsmtFinSF2",
                       "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
                       "2ndFlrSF", "LowQualFinSF", "GrLivArea",  # "GarageYrBlt",
                       "GarageArea", "WoodDeckSF", "OpenPorchSF",
                       "EnclosedPorch", "3SsnPorch", "ScreenPorch",
                       "PoolArea", "MiscVal", "SalePrice"]
    dispersed_dict = [i for i in list(train_df_filled.columns) if i not in continuity_dict]
    i = 1
    for var in vars:
        if var not in continuity_dict:
            le = preprocessing.LabelEncoder()
            le.fit(list(set(train_df_filled[var])))
            train_df_filled[var] = le.transform(train_df_filled[var])
            for label in le.classes_:
                print(label, le.transform([label])[0])

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.boxplot(x=var, y=output, data=train_df_filled)
            ax.set_ylim(0, 800000)
            plt.xticks(rotation=90)
        else:
            fig, ax = plt.subplots()
            ax.scatter(x=train_df_filled[var], y=train_df_filled[output])
            plt.ylabel(output, fontsize=8)
            plt.xlabel(var, fontsize=8)
        plt.savefig(str(i) + ".png")
        plt.close()
        i = i + 1

def show_data():
    train_df, test_df = load_data()
    df = concat_data()
    sns.distplot(train_df['SalePrice'])
    plt.savefig("直方图展示.png")
    print("斜度: %f" % train_df['SalePrice'].skew())
    print("峭度: %f" % train_df['SalePrice'].kurt())
    show_feature(train_df)
    feature_analysis(df, train_df)

#特征分析
def feature_analysis(df,train_df):
    # 1.统计缺失值
    # print(pd.isnull(df).sum())
    # print(df.isnull().sum().sort_values(ascending=False).head(20))
    df_na = (df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
    miss_rate = pd.DataFrame({'缺失率': df_na})
    f, ax = plt.subplots(figsize=(10, 10))
    plt.xticks(rotation='90')
    sns.barplot(x=df_na.index, y=df_na)
    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Percent of missing values', fontsize=10)
    plt.title('Percent missing data by feature', fontsize=10)
    plt.savefig("缺失值统计.png")
    # print(miss_rate.head(10))
    # 2.对每个特征进行相关性分析
    corrmat = train_df.corr()
    plt.subplots(figsize=(15, 12))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.savefig("相关性分析.png")


#剔除缺失值多的特征（缺失率大于10%）
def drop_na_fiture(df):
    df_na = (df.isnull().sum()/len(df))
    df_na = pd.DataFrame({'rate':df_na})
    na_fiture = list(df_na[df_na['rate']>0.1].index)
    df.drop(columns=na_fiture,inplace=True)
    return df

#选择与价格相关性最高的特征
def choose_feature(train_df):
    corrmat = train_df.corr()
    feature = pd.DataFrame({'corr':corrmat['SalePrice'].sort_values(ascending=False)})
    feature.drop('SalePrice',inplace=True)
    my_feature = list(feature[feature['corr']>0.5].index)
    return my_feature
    # cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(train_df[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
    # plt.show()

#对特征进行类型转换
def change_dtype_feature(df,feature_list):
    columns = df.columns
    for column in columns:
        if column in feature_list:
            df[column] = df[column].astype(str)
        elif(df[column].dtype=='int64'):
            df[column] = df[column].astype('float64')

    return df

#查找所有非类别特征（用于特征转换）
def find_nonobject_feature(df):
    cols = df.columns
    for col in cols:
        if(str(df[col].dtype)!='object'):
            print(col,'   ',df[col].dtype)

#缺失值填充
def fill_nan_feature(df):
    fill_data = {}
    columns = df.columns
    for column in columns:
        if (str(df[column].dtype) == 'object'):
            fill_data[column] = df[column].value_counts().index[0]
        else:
            fill_data[column] = df[column].mean()
    mdf = df.fillna(fill_data)
    return mdf

#单个特征归一化
def normalization(data):
    oridata = {}
    oridata['min'] = min(data)
    oridata['max'] = max(data)
    data -= min(data)
    data /= (max(data) - min(data))
    return data, oridata

#数据归一化
def data_normalization(df):
    oridatas = {}
    columns = df.columns
    for column in columns:
        if(str(df[column].dtype)!='object'):
            df[column],oridata = normalization(df[column])
            oridatas[column] = oridata
    return  df,oridatas

#去除孤立点
def drop_isolated_points(train_df,ndf):
    '''
    Drop:
    LotArea > 75000
    MasVnrArea > 14000
    LowQualFinSF
    PoolArea
    3SsnPorch
    MiscVal
    one-hot:
    BsmtHalfBath
    HalfBath
    BsmtFullBath
    FullBath
    BedroomAbvGr
    KitchenAbvGr
    Fireplaces
    TotRmsAbvGrd
    GarageCars
    '''
    sale_price = train_df['SalePrice']
    drop_ind = ndf['LotArea'][ndf['LotArea']>60000].index
    ndf.drop(index=drop_ind, inplace=True)
    train_df.drop(index=drop_ind, inplace=True)
    sale_price.drop(index=drop_ind,inplace=True)
    drop_ind = ndf['MasVnrArea'][ndf['MasVnrArea'] > 1400].index
    train_df.drop(index=drop_ind, inplace=True)
    sale_price.drop(index=drop_ind,inplace=True)
    ndf.drop(index=drop_ind,inplace=True)
    drop_col = ['LowQualFinSF','PoolArea','3SsnPorch','MiscVal']
    ndf.drop(columns=drop_col,inplace=True)

    # change_dtype_feature_list = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold','BsmtHalfBath','HalfBath']
    # ndf = change_dtype_feature(ndf,change_dtype_feature_list)
    # columns = ndf.columns
    # for column in columns:
    #     if str(ndf[column].dtype)!='object':
    #         plt.figure()
    #         plt.plot(range(len(ndf[column])),ndf[column].values,label=column)
    #         plt.legend()
    #         plt.show()

    return ndf,train_df,sale_price

#数据处理
def data_process(df,train_df):
    '''
    1. 去除缺失值多的特征（缺失率）
    2. 进行类型转换。
        find_nonobject_feature(df):
        MSSubClass
        OverallQual
        OverallCond
        YearBuilt
        YearRemodAdd
        GarageYrBlt
        MoSold
        YrSold
        ①以上类型实际为类别但读入为数字型，因此进行类型转换
        ②将int64转化为float64
    (6.更新：去除孤立点)
    3.对缺失值进行填充（category类型采用最多值填充,numerical类型采用均值填充）
    4. 对所有的category类型的值采用one-hot编码(使用函数pd.get_dummies())
    5. 对所有numerical类型的值进行归一化
    '''
    #1.去除缺失率高于10%的特征值
    ndf = drop_na_fiture(df)

    #2.进行类型转换
    change_dtype_feature_list = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']
    change_dtype_feature_list.append('BsmtHalfBath')
    change_dtype_feature_list.append('HalfBath')
    # change_dtype_feature_list.append('BsmtFullBath')
    # change_dtype_feature_list.append('FullBath')
    # change_dtype_feature_list.append('BedroomAbvGr')
    # change_dtype_feature_list.append('KitchenAbvGr')
    # change_dtype_feature_list.append('Fireplaces')
    # change_dtype_feature_list.append('TotRmsAbvGrd')
    # change_dtype_feature_list.append('GarageCars')

    ndf = change_dtype_feature(ndf,change_dtype_feature_list)

    #6.更新-去除孤立点
    ndf, train_df, sale_price = drop_isolated_points(train_df,ndf)
    #3.对缺失值进行填充
    ndf = fill_nan_feature(ndf)

    #4.对所有的category类型的值采用one - hot编码
    ndf = pd.get_dummies(ndf)

    #5.对所有numerical类型的值进行归一化
    ndf,oridatas = data_normalization(ndf)

    return ndf,train_df,sale_price,oridatas

#数据预处理
def data_preprocess():
    train_df, test_df = load_data()
    df = concat_data()
    df,train_df,sale_price,datas = data_process(df,train_df)
    '''
        由于价格的差值比较大，使用函数np.log1p()对训练集价格进行数据平滑处理。
        使其更好的服从高斯分布，便于后续训练和预测。
        最后使用函数np.expm1()对预测的数据进行还原。
    '''
    sale_price = np.log1p(sale_price)
    sale_price,datas['SalePrice'] = normalization(sale_price)
    n_train_df = df.reindex(train_df.index)
    n_test_df = df.reindex(test_df.index)
    return n_train_df,n_test_df,sale_price,datas