# 根据图片生成静态视频
import os
import cv2


def resize_images(image_list, target_size):
    resized_images = []
    for image_path in image_list:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return resized_images


def images_to_video(image_list, output_video, duration_per_image):
    # 获取第一张图片的宽度和高度
    # img = cv2.imread(image_list[0])
    height, width, _ = image_list[0].shape

    # 设置视频编解码器和视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4v 编解码器
    out = cv2.VideoWriter(
        output_video, fourcc, 0.2, (width, height)
    )  # 帧率：每秒显示的图像个数

    # 计算每张图片需要写入视频的次数
    total_frames_per_image = int(duration_per_image * 0.2)  # 每张图片需要显示的总帧数

    # 遍历图片列表，将每张图片重复写入视频
    for image in image_list:
        for _ in range(total_frames_per_image):
            out.write(image)

    # 释放视频写入对象
    out.release()


if __name__ == "__main__":
    # 图片目录
    image_dir = "test"

    # 获取图片列表
    image_list = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]
    print(len(image_list))

    image_list = resize_images(image_list, (480, 480))

    # 输出视频路径
    output_video = "utils\output.mp4"

    # 每张图片占据的时间（秒）
    duration_per_image = 5

    # 调用函数生成视频
    images_to_video(image_list, output_video, duration_per_image)
