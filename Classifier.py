# 用于分类器的构建
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

# 用于保存模型
import joblib

# 四种分类器
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

# 获取特征提取函数
from Feature_extraction import get_cnn_features


# 根据模型路径加载模型
def initialize_model(model_path):
    model = joblib.load(model_path)  # 加载模型
    return model


# 根据加载的模型在单个图片上进行预测
def predict_module(image_path):

    model = initialize_model("model_saved\\best_model.pkl")  # 加载模型

    image_class = {0: "AI", 1: "Real"}

    temp_data = get_cnn_features(image_path)  # 获取特征选择后的数据

    predictions_labels = int(model.predict(temp_data))
    prob = list(model.predict_proba(temp_data)[0])  # 返回输入样本的每个类别的概率估计值
    res_dict = dict()
    res_dict["pred_label"] = predictions_labels
    res_dict["pred_score"] = prob[predictions_labels]
    res_dict["pred_class"] = image_class[predictions_labels]

    # 读取原始图片
    image = cv2.imread(image_path[0])

    # 在图片上绘制矩形框
    cv2.rectangle(image, (5, 5), (450, 40), (100, 100, 100), -1)  # 绘制黑色矩形框

    # 在图片上绘制预测结果
    text = f"Predicted Label: {res_dict['pred_label']}, Predicted Score: {res_dict['pred_score']:.2f}"
    cv2.putText(
        image, text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1
    )
    # 保存带有预测结果的图片
    output_image_path = "temp_picture\\predicted_image.png"
    cv2.imwrite(output_image_path, image)

    return res_dict


def plot_loss(clf, X, y):
    # 记录迭代次数和损失函数值
    num_iterations = []
    losses = []

    # 迭代训练模型，并记录损失函数值
    for i in range(1, 1001):  # 设定最大迭代次数为1000
        clf.set_params(max_iter=i)
        clf.fit(X, y)
        num_iterations.append(i)
        losses.append(1 - clf.score(X, y))  # 使用误分类率作为损失函数，即 1 - 准确率

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(num_iterations, losses, color="blue")
    plt.title("SVM Training Process")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss (Misclassification Rate)")
    plt.grid(True)
    plt.show()


def main():

    mpl.rcParams["font.sans-serif"] = ["KaiTi"]  # 设置无衬线字体为楷体
    mpl.rcParams["font.serif"] = ["KaiTi"]  # 设置有衬线字体为楷体

    # ----------------------------------------------------------------------------------
    # 第一步 划分训练集和测试集
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

    # ----------------------------------------------------------------------------------
    # 第二步 图像读取及转换为像素直方图并进行特征选择
    # ----------------------------------------------------------------------------------

    XX_train = get_cnn_features(X_train)  # 训练集
    XX_test = get_cnn_features(X_test)  # 测试集

    # ----------------------------------------------------------------------------------
    # 第三步 分别选择支持向量机、决策树、KNN、朴素贝叶斯创建分类器
    # ----------------------------------------------------------------------------------
    basic_model = "SVM"
    models = {
        "SVM": SVC(
            kernel="rbf",  # 常见核函数包括‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
            probability=True,  # 启用概率估计
            C=5,
            gamma=0.005,
        ),  # 此处为网格搜索所确定的最优超参数
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "BernoulliNB": BernoulliNB(),
    }

    select_model = models[basic_model]

    plot_loss(select_model, XX_train, y_train)

    clf = select_model.fit(XX_train, y_train)

    # 保存模型
    joblib.dump(clf, "model_saved\\best_model2.pkl")

    predictions_labels_tr = clf.predict(XX_train)
    print("训练集上的算法评价:")
    print(
        classification_report(y_train, predictions_labels_tr)
    )  # 生成分类模型的详细性能报告

    predictions_labels = clf.predict(XX_test)

    print("真实标签为：")
    print(y_test)
    print("预测结果:")
    print(predictions_labels)
    print("算法评价:")
    print(
        classification_report(y_test, predictions_labels)
    )  # 生成分类模型的详细性能报告

    y_true = y_test  # 正确标签
    y_pred = predictions_labels  # 预测标签
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    np.set_printoptions(precision=2)  # 设置numpy数组的打印选项，以显示2位小数的浮点值
    cm_normalized = (
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    )  # 将混淆矩阵进行归一化

    # 调用绘制混淆矩阵的函数，并设置标题
    plot_confusion_matrix(cm_normalized, title="Normalized confusion matrix")
    # 保存混淆矩阵图像
    plt.savefig("Result\Train\matrix.png", format="png")
    # 显示混淆矩阵图像
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, title="Confusion Matrix", cmap=plt.cm.binary):
    # 二分类，所以有两个类别
    labels = [0, 1]

    plt.figure(figsize=(12, 8), dpi=120)  # 设置绘图的图形大小

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)  # 创建坐标网格

    # 遍历每个单元格，并添加文本标签
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # 只对大于0.01的单元格添加标签
        if c > 0.01:
            plt.text(
                x_val,
                y_val,
                "%0.2f" % (c,),  # 显示两位小数
                color="red",  # 标签颜色为红色
                fontsize=30,  # 标签字体大小为7
                fontweight="bold",  # 加粗文本
                va="center",  # 垂直方向居中对齐
                ha="center",  # 水平方向居中对齐
            )
    tick_marks = np.array(range(len(labels))) + 0.5  # 为了使得标签在中间显示
    # 调整刻度以使其更清晰
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position("none")  # 隐藏x轴刻度
    plt.gca().yaxis.set_ticks_position("none")  # 隐藏y轴刻度
    plt.grid(True, which="minor", linestyle="-")  # 显示网格线

    plt.gcf().subplots_adjust(bottom=0.15)  # 调整底部边距

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


if __name__ == "__main__":
    main()
