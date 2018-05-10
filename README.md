# cv-homework

实验环境

1、改写test文件目录下的路径，
deploy.prototxt(文件夹中中vgg_deploy.prototxt) caffemodel（http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel ） 
mean.binaryproto（文件夹中提供） 
synset_words.txt（将categories_places365.txt中对应的类别提取出来并排序）图片用自己选择的测试图片

2、运行caffe：
修改test.txt中的代码

'''caffe.exe deploy.prototxt vgg16_places365.caffemodel mean.binaryproto synset_words.txt 111.jpg'''
pause 
