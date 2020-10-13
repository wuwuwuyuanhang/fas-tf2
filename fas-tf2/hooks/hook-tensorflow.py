# @Auther : wuwuwu 
# @Time : 2020/8/25 
# @File : hook-tensorflow.py
# @Description :

#从pyinstaller库中引用了模块
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

#下面用于将以下模块下所有的子模块全部加入
hiddenimports = collect_submodules('tensorflow_core.core.framework')
hiddenimports += collect_submodules('tensorflow_core.core')
hiddenimports += collect_submodules('tensorflow_core')
hiddenimports += collect_submodules('tensorflow_core.lite.experimental.microfrontend.python.ops')
