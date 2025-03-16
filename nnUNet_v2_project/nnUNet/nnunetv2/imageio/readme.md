- Derive your adapter from `BaseReaderWriter`. 
- Reimplement all abstractmethods. 
- make sure to support 2d and 3d input images (or raise some error).
- place it in this folder or nnU-Net won't find it!
- add it to LIST_OF_IO_CLASSES in `reader_writer_registry.py`

从 BaseReaderWriter 派生dataset的适配器，步骤：
-重新实现所有抽象方法。
-确保支持 2D 和 3D 输入图像（或引发错误）。
-将其放置在此文件夹中，否则 nnU-Net 将无法找到它！
-将其添加到 reader_writer_registry.py 中的 LIST_OF_IO_CLASSES。
完成！