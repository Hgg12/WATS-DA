import cv2
import os

# 主文件夹路径，其中包含多个子文件夹
base_folder = r"E:\WATB\WATB"
algorithm2_path = r"E:\WATB_Benchmark_toolkit\results\WATB\WATS-DA(RBO)_tracking_result"
algorithm3_path = r"E:\WATB_Benchmark_toolkit\results\WATB\WATS-DA(CAR)_tracking_result"
algorithm4_path = r"E:\WATB_Benchmark_toolkit\results\WATB\SiamRBO_tracking_result"
algorithm5_path = r"E:\WATB_Benchmark_toolkit\results\WATB\SiamCAR_tracking_result"
output_base_path = r"F:\WATB"

# 遍历主文件夹下的所有子文件夹
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
     # 仅处理文件夹
    if os.path.isdir(folder_path):
        print(f"正在处理文件夹: {folder_name}")
        # 各个算法的groundtruth文件路径
        data_paths = {
            "algorithm1":  os.path.join(folder_path, "groundtruth_rect.txt"),  # groundtruth
            "algorithm2":  os.path.join(algorithm2_path, f"{folder_name}.txt"),  # WATS-DA(RBO)的groundtruth
            "algorithm3":  os.path.join(algorithm3_path, f"{folder_name}.txt"),   # WATS-DA(CAR)的groundtruth
            "algorithm4":  os.path.join(algorithm4_path, f"{folder_name}.txt"),   # SiamRBO的groundtruth
            "algorithm5": os.path.join(algorithm5_path, f"{folder_name}.txt")    # SiamCAR的groundtruth
        }

        image_path = os.path.join(folder_path, "img")   # 图片所在目录
        image_path2 = os.path.join(output_base_path, folder_name)   # 保存处理后图片的目录

        # 自动创建保存图片的目录
        if not os.path.exists(image_path2):
            os.makedirs(image_path2)

        # 定义不同算法对应的颜色，每个算法的结果用不同颜色表示
        colors = {
            "algorithm1": (0, 255, 0),    # 绿色
            "algorithm2": (0, 0, 255),    # 红色
            "algorithm3": (255, 0, 0),    # 蓝色
            "algorithm4": (0, 255, 255),  # 黄色
            "algorithm5": (255, 0, 255)   # 紫色
        }

        # 读取所有算法的groundtruth文件
        groundtruth_data = {}
        for algorithm, path in data_paths.items():
            with open(path) as fb:
                groundtruth_data[algorithm] = fb.readlines()  # 每个算法的groundtruth数据存入字典
        # 获取实际图片数量
        image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
        # 对图片和数据进行循环处理
        # for i in range(1, image_files):  # 遍历图片编号，从1到9999
        #     image_path1 = image_path + "\\" + str(i).zfill(4) + ".jpg"  # 图片的绝对路径
        #     image_path3 = image_path2 + "\\" + str(i).zfill(4) + ".jpg"  # 保存处理后的图片路径
        # 对图片和数据进行循环处理
        for i, image_file in enumerate(image_files, 1):
            image_path1 = os.path.join(image_path, image_file)  # 图片的绝对路径
            image_path3 = os.path.join(image_path2, image_file)  # 保存处理后的图片路径
            
            image = cv2.imread(image_path1)  # 读取图片

            if image is None:
                print(f"无法读取图片: {image_path1}")
                continue  # 如果图片读取失败，跳过
            
             # 在图片左上角标注帧编号（第几帧）
            frame_number_text = f"Frame: {i}"
            cv2.putText(image, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 遍历每个算法的groundtruth并绘制不同颜色的方框
            for algorithm, lines in groundtruth_data.items():
                value = lines[i-1].split(',')  # 按“，”分割groundtruth数据
                # 每四个值为一组groundtruth (x_min, y_min, width, height)
                for j in range(0, len(value), 4):
                    result = [v.rstrip() for v in value[j:j+4]]  # 每四个值为一组groundtruth
                    x_min = float(result[0])  # 左上角x坐标
                    y_min = float(result[1])  # 左上角y坐标
                    x_max = float(result[0]) + float(result[2])  # 右下角x坐标（x_min + 宽度）
                    y_max = float(result[1]) + float(result[3])  # 右下角y坐标（y_min + 高度）
                    
                    # 确保坐标是整数
                    first_point = (int(round(x_min)), int(round(y_min)))  # 方框左上角点
                    last_point = (int(round(x_max)), int(round(y_max)))  # 方框右下角点
                    
                    # 绘制不同算法的groundtruth方框，颜色依据算法选择
                    cv2.rectangle(image, first_point, last_point, colors[algorithm], 2)  # 绘制矩形框
            
            # 将绘制了多个groundtruth方框的图片保存
            cv2.imwrite(image_path3, image)
