def get_my_monai_dir(user_name):
    import os
    path = rf'C:\Users\{user_name}\Data\Monai_data_dir'
    if os.path.exists(path):
        return os.path.normpath(path)