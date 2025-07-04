import os

folder_path = r'D:\BaiduNetdiskDownload\huaxi\1945.4.1-1945.12.31'  # 请替换为你的文件夹路径

for folder, subfolders, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(folder, filename)
        # 如果文件后缀为.jpeg，则将其改为.jpg
        if filename.endswith('.jpeg'):
            new_filename = filename.rsplit('.', 1)[0] + '.jpg'
            new_file_path = os.path.join(folder, new_filename)
            os.rename(file_path, new_file_path)
        # 如果文件后缀为.livp，则将其删除
        elif filename.endswith('.livp'):
            os.remove(file_path)


# for folder, subfolders, files in os.walk(folder_path):
#     for filename in files:
#         file_path = os.path.join(folder, filename)
#         # 如果文件后缀为.livp.jpg，则将其改为.jpg
#         if filename.endswith('.livp.jpg'):
#             new_filename = filename.rsplit('.', 2)[0] + '.jpg'
#             new_file_path = os.path.join(folder, new_filename)
#             os.rename(file_path, new_file_path)