# 用于数据集的重新编码
import os
import cv2


def convert_and_rename(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹下所有文件
    files = os.listdir(folder_path)
    files.sort()  # 按名称排序

    # 重命名并转换文件格式
    count = 0
    for file_name in files:
        # 获取文件的完整路径
        old_path = os.path.join(folder_path, file_name)

        # 检查文件扩展名是否为 '.jpg'
        if not file_name.lower().endswith(".jpg"):
            # 读取文件
            image = cv2.imread(old_path)

            # 构造新的文件名（包括扩展名）
            new_name = str(count + 1) + ".jpg"
            new_path = os.path.join(folder_path, new_name)

            # 将图片保存为 JPG 格式
            cv2.imwrite(new_path, image)

            # 删除原始非 JPG 格式图片
            os.remove(old_path)
        else:
            # 如果文件已经是 JPG 格式，只重命名
            new_name = str(count + 1) + ".jpg"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)

        count += 1


# 要重命名并转换格式的文件夹路径
folder_path = "datasets\Al_from_kaggle\Real"

# 调用函数进行重命名和转换格式
convert_and_rename(folder_path)
