import os
import shutil


def clear_dir(list_dir):
    for i in list_dir:
        for filename in os.listdir(i):
            file_path = os.path.join(i, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    pass
