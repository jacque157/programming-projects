import os
import shutil

input_path = os.path.join('CMU')
output_path = os.path.join('CMU_lite')


for file in os.listdir(input_path):
    path1 = os.path.join(input_path, file)
    out_path1 = os.path.join(output_path, file)
    if os.path.isdir(path1):
        os.mkdir(out_path1)
        for file2 in os.listdir(path1):
            path2 = os.path.join(path1, file2)
            out_path2 = os.path.join(out_path1, file2)
            if not os.path.isdir(path2):
                with open(out_path2, 'w') as f:
                    pass
                shutil.copyfile(path2, out_path2)

    else:
        with open(out_path1, 'w') as f:
            pass
        shutil.copyfile(path1, out_path1)
