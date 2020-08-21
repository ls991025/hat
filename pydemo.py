# -*- coding: gbk -*-

from flask import request, Flask
import json
import base64
import cv2
import numpy as np
from gluoncv import data, utils
from mxnet import gluon
import mxnet as mx
import requests
#导入多个模块
classes = ['hat', 'person']
#定义类型
ctx = mx.cpu()
#采用cpu方式进行识别

url = ''#填入接口路径
r = requests.get(url)
result = json.loads(r.text)
base2=result["data"]#取出接口信息

app = Flask(__name__)
@app.route("/", methods=['POST','GET'])
def get_frame():
 hnum=0
 pnum=0
 i=0
 frame="9.jpg"
 img = cv2.imread(frame)#以cv2方式打开目的图片
 x, img = data.transforms.presets.yolo.load_test(frame, short=416)
 x = x.as_in_context(ctx)
 net=gluon.SymbolBlock.imports(symbol_file='./darknet53-symbol.json', input_names=['data'], param_file='./darknet53-0000.params', ctx=ctx)
 box_ids, scores, bboxes = net(x)
 ax = utils.viz.cv_plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=classes,thresh=0.4)
 #对图片进行识别
 #根据属性对类别数量分别进行统计
 while i < 100:
  if (box_ids[0][i]==0)and(scores[0][i]>0.4):
   hnum=hnum+1
  elif (box_ids[0][i]==1)and(scores[0][i]>0.4):
   pnum=pnum+1
  else:
   break
  i=i+1
 else:
   print ("exit")
 #将数量信息在结果图片中显示
 cv2.putText(img, classes[0]+"="+str(hnum), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
 cv2.putText(img, classes[1]+"="+str(pnum), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
 cv2.imwrite(frame.split('.')[0] + '_result.jpg', img[...,::-1])
 cv2.destroyAllWindows()
 with open(frame.split('.')[0] + '_result.jpg', "rb") as f:
    # b64encode：编码，b64decode: 解码
    base64_data = base64.b64encode(f.read())
    base=str(base64_data,'utf8') 
    # base64.b64decode(base64_data)
    context={
                   'h':hnum,
                   'p':pnum,
                   'data':base
}
    #返回context信息回接口 
    json_str = json.dumps(context)
    #return str(context)
    return json_str
if __name__ == "__main__":
    app.run("127.0.0.1", port=5000)