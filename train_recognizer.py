# USAGE
# python train_recognizer.py --checkpoints fer2013/checkpoints
# python train_recognizer.py --checkpoints fer2013/checkpoints --model fer2013/checkpoints/epoch_20.hdf5 \
#	--start-epoch 20

# set the matplotlib backend so figures can be saved in the background
# 导入工具包

import matplotlib
matplotlib.use("Agg")

#这段代码导入了一些必须要用到的包：
#包括配置文件、图像预处理、回调函数、数据集生成器、VGG网络模型、图像数据生成器、优化器、模型加载器等。
# import the necessary packages
from config import emotion_config as config
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.callbacks import EpochCheckpoint
from pyimage.callbacks import TrainingMonitor
from pyimage.io import HDF5DatasetGenerator
from pyimage.nn.conv.emotionvggnet import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os

# construct the argument parse and parse the arguments
# 命令行参数
'''
创建了一个解析命令行参数的工具。通过这个工具，从命令行传递参数，告诉程序：

保存模型权重的目录 (checkpoints)。
要加载的特定模型文件 (model)。
从哪个epoch开始训练 (start-epoch)。
'''
ap = argparse.ArgumentParser()
# checkpoint:在网络训练过程中将权重进行保存
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
# 指定获取哪个具体的checkpoint
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
# 指定当前开始训练的epoch
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
# 图像增强
'''
创建了两个图像数据增强器：

trainAug：用于训练数据，做一些随机变换如旋转、缩放、水平翻转等，以增加数据的多样性。
valAug：用于验证数据，只做归一化处理。
此外，还初始化了一个图像转数组的预处理器 (iap)，将图像转换成适合模型输入的格式。
'''
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
	horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
# 实例化
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
# 得到训练和验证的数据的生成器
'''
这里，创建了两个数据集生成器：

trainGen：用于训练数据，从HDF5文件读取数据，并应用之前定义的数据增强和预处理。
valGen：用于验证数据，同样从HDF5文件读取数据，并应用预处理。
'''
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
	aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
# 未指定具体的checkpoint，从头训练
'''
如果没有指定模型文件，会创建一个新的模型：

使用EmotionVGGNet构建模型，指定图像的宽、高、深度（通道数）和分类数量。
使用Adam优化器，学习率设置为1e-3。
编译模型，指定损失函数为categorical_crossentropy，评价指标为accuracy。
'''
if args["model"] is None:
	print("[INFO] compiling model...")
	model = EmotionVGGNet.build(width=48, height=48, depth=1,
		classes=config.NUM_CLASSES)
	opt = Adam(lr=1e-3)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# 指定了具体的checkpoint
'''
如果指定了模型文件，会加载这个模型，并且：

打印当前的学习率。
将学习率设置为1e-5。
再次打印新的学习率
'''
else:
	print("[INFO] loading {}...".format(args["model"]))
	# 加载checkpoint
	model = load_model(args["model"])

	# 获取当前参数
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# 定义回调函数
'''
这里定义了两个回调函数：

EpochCheckpoint：每隔5个epoch保存一次模型权重。
TrainingMonitor：记录训练过程，并将结果保存为图像和JSON文件。
'''
figPath = os.path.sep.join([config.OUTPUT_PATH,
	"vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,
	"vggnet_emotion.json"])
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5,
		startAt=args["start_epoch"]),
	TrainingMonitor(figPath, jsonPath=jsonPath,
		startAt=args["start_epoch"])]

# 训练网络
'''
我们开始训练模型：

使用训练数据生成器生成的数据。
每个epoch进行的训练步数等于训练样本总数除以批量大小。
使用验证数据生成器生成的数据进行验证。
每个epoch进行的验证步数等于验证样本总数除以批量大小。
总共训练15个epoch。
max_queue_size设置为10，表示生成器队列的最大尺寸。
使用之前定义的回调函数。
打印详细的训练过程信息。

'''
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=15,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# 关闭数据流
trainGen.close()
valGen.close()