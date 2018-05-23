#!/usr/bin/env python--将mean.binaryproto文件转为python可以使用的mean.npy文件
import numpy as np
import sys,caffe
 
root='/home/51744/cv-homework/'  #设置根目录
mean_proto_path=root+'places365CNN_mean.binaryproto'    #mean.binaryproto路径
mean_npy_path=root+'mean.npy'              #mean.npy路径
 
blob=caffe.proto.caffe_pb2.BlobProto()     #创建protobuf blob
data=open(mean_proto_path,'rb').read()     #读入mean.binaryproto文件内容
blob.ParseFromString(data)                 #解析文件内容到blob
 
array=np.array(caffe.io.blobproto_to_array(blob))  #将blob中的均值转换称numpy格式，array的shape(mean_number,channel,hight,width)
mean_npy=array[0]                          #一个array中可以有多组均值存在，故需要通过下标选择一组均值
np.save(mean_npy_path,mean_npy)            #保存