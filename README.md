光伏面板缺陷检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ef28ed9c8ee04e548eaad0346cfc2892.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/501567c565b348688746ba1b37c3f9e5.png)
3636张，yolo和voc两种标注方式
6类，标注数量：
Defective-缺陷: 2090
Bird Drop-鸟粪: 2217
Snow-积雪: 2185
Non Defective-无缺陷: 2273
Dust-尘土: 2185
Physical Damage-物理损坏: 2302
image num: 3636![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0986d67d9a5b43299f512469016a0617.png)


2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示，附全套代码和训练好的权重文件。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe8aa44fb8824154b54ed10d7e40199a.png)



3.qt界面：可视化运行界面采用pyqt编写，可视化界面简单友好，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ceca94b41d2943d69e6d600a2c0fb0f4.png)




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/522f29e061064212a1809405d45f1d78.png)


)
```


# YOLOv11光伏面板污损检测完整工程代码
## pv.yaml
```yaml
path: ./datasets/pv
train: images/train
val: images/val
test: images/test
names:
  0: Defective
  1: Bird Drop
  2: Snow
  3: Non Defective
  4: Dust
  5: Physical Damage
```
## train.py
```python
from ultralytics import YOLO
model=YOLO("yolo11n.pt")
model.train(data="pv.yaml",epochs=80,imgsz=640,batch=16,device=0)
```
## PyQt5界面
```python
import sys,cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt,QTimer
from ultralytics import YOLO
class PvDetect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的光伏面板污损检测系统")
        self.resize(1250,820)
        self.model=YOLO("runs/train/pv/weights/best.pt")
        self.cap=None
        self.timer=QTimer()
        self.timer.timeout.connect(self.frame)
        self.ui()
    def ui(self):
        w=QWidget()
        self.setCentralWidget(w)
        lay=QGridLayout(w)
        self.imgLab=QLabel("图像区")
        self.imgLab.setFixedSize(640,640)
        self.imgLab.setStyleSheet("border:1px solid #666")
        lay.addWidget(self.imgLab,0,0,4,1)
        lay.addWidget(QPushButton("图片检测",clicked=self.imgDet),0,1)
        lay.addWidget(QPushButton("视频检测",clicked=self.vidDet),1,1)
        lay.addWidget(QPushButton("摄像头",clicked=self.openCam),2,1)
        self.info=QTextEdit()
        self.info.setReadOnly(True)
        lay.addWidget(self.info,3,1)
        self.table=QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(["序号","路径","类别","置信度","坐标"])
        lay.addWidget(self.table,4,0,1,2)
    def imgDet(self):
        p,_=QFileDialog.getOpenFileName()
        if not p:return
        res=self.model.predict(p,conf=0.5)[0]
        img=cv2.imread(p)
        self.table.setRowCount(0)
        for i,box in enumerate(res.boxes):
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cls=self.model.names[int(box.cls[0])]
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f"{cls} {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            r=self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r,0,QTableWidgetItem(str(i+1)))
            self.table.setItem(r,1,QTableWidgetItem(p))
            self.table.setItem(r,2,QTableWidgetItem(cls))
            self.table.setItem(r,3,QTableWidgetItem(f"{conf*100:.2f}%"))
            self.table.setItem(r,4,QTableWidgetItem(f"[{x1},{y1},{x2},{y2}]"))
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        hh,ww,c=rgb.shape
        qim=QImage(rgb.data,ww,hh,c*ww,QImage.Format_RGB888)
        self.imgLab.setPixmap(QPixmap.fromImage(qim).scaled(640,640,Qt.KeepAspectRatio))
        self.info.append(f"检测目标总数:{len(res.boxes)}")
    def vidDet(self):
        f,_=QFileDialog.getOpenFileName()
        if not f:return
        self.cap=cv2.VideoCapture(f)
        self.timer.start(30)
    def openCam(self):
        self.cap=cv2.VideoCapture(0)
        self.timer.start(30)
    def frame(self):
        ret,frm=self.cap.read()
        if not ret:
            self.timer.stop()
            return
        res=self.model.predict(frm,conf=0.5)[0]
        for box in res.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cls=self.model.names[int(box.cls[0])]
            cv2.rectangle(frm,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frm,f"{cls} {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        rgb=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
        hh,ww,c=rgb.shape
        qim=QImage(rgb.data,ww,hh,c*ww,QImage.Format_RGB888)
        self.imgLab.setPixmap(QPixmap.fromImage(qim).scaled(640,640,Qt.KeepAspectRatio))
if __name__=="__main__":
    app=QApplication(sys.argv)
    win=PvDetect()
    win.show()
    sys.exit(app.exec_())
```
依赖安装
```bash
pip install ultralytics opencv-python pyqt5
```
