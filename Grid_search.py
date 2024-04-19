# 执行网格搜索以获得最优超参数
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import seaborn as sns
from Feature_extraction import get_cnn_features


# 绘制网格搜索热力图
def visualize_heatmap(param_grid, results):
    # 从参数网格中提取gamma和C值
    gammas = np.array(param_grid["gamma"])
    Cs = np.array(param_grid["C"])

    # 重塑平均训练和测试分数以匹配网格形状
    mean_train_scores = np.array(results["mean_train_score"]).reshape(
        len(Cs), len(gammas)
    )
    mean_test_scores = np.array(results["mean_test_score"]).reshape(
        len(Cs), len(gammas)
    )

    # 绘制mean train scores heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        mean_train_scores,
        annot=True,
        fmt=".4f",
        xticklabels=gammas,
        yticklabels=Cs,
        cmap="viridis",
    )
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Mean Train Scores Heatmap")
    plt.show()

    # 绘制mean test scores heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        mean_test_scores,
        annot=True,
        fmt=".4f",
        xticklabels=gammas,
        yticklabels=Cs,
        cmap="viridis",
    )
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Mean Test Scores Heatmap")
    plt.show()


# ----------------------------------------------------------------------------------
# 第一步 获取、划分、归一化数据集用于网格搜索
# ----------------------------------------------------------------------------------

X = []  # 定义图像名称
Y = []  # 定义图像分类类标

for i in range(0, 2):
    # 遍历文件夹，读取图片
    if i == 0:
        dir = "AI"
    else:
        dir = "Real"

    for f in os.listdir(f"datasets/Al_from_kaggle/{dir}"):
        # 获取图像名称
        X.append("datasets/Al_from_kaggle/" + dir + "//" + str(f))
        # 获取图像类标即为文件夹名称
        Y.append(i)

X = np.array(X)
Y = np.array(Y)

# 随机率为100% 选取其中的20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=1, stratify=Y
)

print("数据集的大小为: ", len(X_train), len(X_test), len(y_train), len(y_test))

XX_train = get_cnn_features(X_train)  # 训练集
XX_test = get_cnn_features(X_test)  # 测试集

# ----------------------------------------------------------------------------------
# 第二步 定义要调优的参数网格、创建GridSearchCV对象
# ----------------------------------------------------------------------------------

# param_grid = {
#     "probability": [True],
#     "C": [0.01, 0.1, 1, 10, 50, 100, 150, 200, 500, 1000, 10000],
#     "gamma": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
#     "kernel": ["rbf"],
# }


param_grid = {
    "probability": [True],
    "C": [1, 5, 10, 15, 20],
    "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "kernel": ["rbf"],
}

# 创建SVM模型
svm = SVC()

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    return_train_score=True,
)

# ----------------------------------------------------------------------------------
# 第三步 在数据集上执行网格搜索并打印网格搜索结果，可视化网格搜索热力图矩阵
# ----------------------------------------------------------------------------------

grid_search.fit(XX_train, y_train)

# 获取cv_results_属性
results = grid_search.cv_results_

# 打印每次网格搜索的参数设置及其在训练集和测试集上的详细得分情况
for mean_train_score, mean_test_score, params in zip(
    results["mean_train_score"], results["mean_test_score"], results["params"]
):
    print("参数设置：", params)
    print("训练集平均得分：", mean_train_score)
    print("测试集平均得分：", mean_test_score)
    print()

# 输出最佳参数组合和对应的得分
print("最佳参数组合:", grid_search.best_params_)
print("最佳模型得分:", grid_search.best_score_)

visualize_heatmap(param_grid, results)
