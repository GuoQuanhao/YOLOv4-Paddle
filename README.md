# YOLOv4-Paddle

A YOLOv4 reproduction by PaddlePaddle

## 数据文件准备

数据集已挂载至aistudio项目中，如果需要本地训练可以从这里下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/105347)，和[标签](https://aistudio.baidu.com/aistudio/datasetdetail/103218)文件

数据集目录大致如下，可根据实际情况修改
```
Data
|-- coco
|   |-- annotions
|   |-- images
|      |-- train2017
|      |-- val2017
|      |-- test2017
|   |-- labels
|      |-- train2017
|      |-- val2017
|      |-- train2017.cache(初始解压可删除，训练时会自动生成)
|      |-- val2017.cache(初始解压可删除，训练时会自动生成)
|   |-- test-dev2017.txt
|   |-- val2017.txt
|   |-- train2017.txt
```

## 训练

### 单卡训练
```
python train.py --batch-size 16 --img 416 416 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp --notest
```
![](https://ai-studio-static-online.cdn.bcebos.com/daf7a96baa614de89e623a0333a1b7d191977db407e64adda171d507b06fb5f9)


### 多卡训练
```
python train_multi_gpu.py --batch-size 32 --img 416 416 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg  --weights '' --name yolov4-pacsp --notest
```
多卡训练项目已提交至[脚本任务YOLOv4](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2337633)

多卡训练日志可在[此处](https://pan.baidu.com/s/1AOzdZTa-kTtuc5f1QhlCdQ)下载，提取码：0cxk


### test-dev数据集验证
```
python testdev.py --img 416 --conf 0.001 --batch 32 --data coco.yaml --cfg cfg/yolov4-mish-416.cfg --weights weights/yolov4-mish-416.weights
```
完成后会生成`detections_test-dev2017_yolov4_results.json`文件，你需要将其压缩为`detections_test-dev2017_yolov4_results.zip`并在[COCO Detection Challenge网站](https://competitions.codalab.org/competitions/20794#participate)提交

![](https://ai-studio-static-online.cdn.bcebos.com/f2a7c0bdffb648b78fa72105d3536142b9c851755ec1435aa6abb3b3bd244056)

提交完成后等待验证结束，点击`View scoring output log`即可下载stdout.txt并查看验证情况
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.622
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754
```

#### 验证结果如下所示

| Model |Frame| Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |  :-: |
| **YOLOv4**<sub>mish-416</sub> |Paddle| 416 | 0.413 | 0.622 | 0.453 | 0.203 | 0.450 | 0.565 | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4-mish-416.cfg) | [weights](https://pan.baidu.com/s/1XMWfVJOAHsFUCsU8pwEr_Q)/r2nw |
| **YOLOv4**<sub>leaky-416</sub> |Paddle| 416 | 0.405 | 0.616 | 0.443 | 0.195 | 0.441 | 0.552 | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4-leaky-416.cfg) | [weights](https://pan.baidu.com/s/1Ppm4654qEgWyoFqTQ--5gQ)/wx7w |
| **YOLOv4** |Paddle| 416 | 0.409 | 0.614 | 0.447 | 0.188 | 0.449| 0.572 | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4.cfg) | [weights](https://pan.baidu.com/s/1Bwoi5nFIrl45T0pKByd_KA)/lwdy |
| **YOLOv4**<sub>mish-416</sub> |Darknet| 416 | 0.415 | 0.633 | 0.447 | 0.219 | 0.444 | 0.553 | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4-pacsp-x-mish.cfg) | [weights](https://drive.google.com/open?id=1NuYL-MBKU0ko0dwsvnCx6vUr7970XSAR) |
| **YOLOv4**<sub>leaky-416</sub> |Darknet| 416 | 0.407 | 0.627 | 0.439 | 0.214 | 0.437 | 0.540 | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4-pacsp-x-mish.cfg) | [weights](https://drive.google.com/open?id=1bV4RyU_-PNB78G-OtoTmw1Q7t_q90GKY) |
| **YOLOv4** |Darknet| 416 | 0.412 | - | - | - | - | - | [cfg](https://github.com/GuoQuanhao/YOLOv4-Paddle/tree/main/cfg/yolov4-pacsp-s.cfg) | [weights](https://drive.google.com/open?id=1L-SO373Udc9tPz5yLkgti5IAXFboVhUt)|

**验证所产生的json文件可在此处下载[yolov4-mish-416](https://pan.baidu.com/s/1Y8Pd4gUuXblGY6C9tYwMVw)/rmb5，[yolov4-leaky-416](https://pan.baidu.com/s/1crmTC3yc768EVCle2DheJg)/nkfo，[darknet](https://pan.baidu.com/s/1mBs-Bs1D-rCzEabEIvX9uw)/lww5**

### 推理

```
python detect.py --cfg cfg/yolov4-pacsp-x.cfg --weights weights/yolov4-pacsp-x.weights
```
运行结果将会保存在inference/output文件夹下

<img src="https://ai-studio-static-online.cdn.bcebos.com/c44854745ee645eda2b29adb74c7236ef6a243a6f14447fdb0262bf960121138" width="300"/><img src="https://ai-studio-static-online.cdn.bcebos.com/00dc6c2592d34c4ead7501297cf8317edd8c5a3963e6455f82d124a3ea1836bf" width="300"/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/78439e72fbd847f4b2948bd1252068a075e6eabbedd446a697fb42598d553aa2" width="300"/><img src="https://ai-studio-static-online.cdn.bcebos.com/478d29f5bf994856884a49f6d324cb48710d4a93c916481c85241f93f53d46a4" width="300"/>



#### [GitHub地址](https://github.com/GuoQuanhao/YOLOv4-Paddle)

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
