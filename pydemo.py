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
#������ģ��
classes = ['hat', 'person']
#��������
ctx = mx.cpu()
#����cpu��ʽ����ʶ��

url = ''#����ӿ�·��
r = requests.get(url)
result = json.loads(r.text)
base2=result["data"]#ȡ���ӿ���Ϣ

app = Flask(__name__)
@app.route("/", methods=['POST','GET'])
def get_frame():
 hnum=0
 pnum=0
 i=0
 frame="9.jpg"
 img = cv2.imread(frame)#��cv2��ʽ��Ŀ��ͼƬ
 x, img = data.transforms.presets.yolo.load_test(frame, short=416)
 x = x.as_in_context(ctx)
 net=gluon.SymbolBlock.imports(symbol_file='./darknet53-symbol.json', input_names=['data'], param_file='./darknet53-0000.params', ctx=ctx)
 box_ids, scores, bboxes = net(x)
 ax = utils.viz.cv_plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=classes,thresh=0.4)
 #��ͼƬ����ʶ��
 #�������Զ���������ֱ����ͳ��
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
 #��������Ϣ�ڽ��ͼƬ����ʾ
 cv2.putText(img, classes[0]+"="+str(hnum), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
 cv2.putText(img, classes[1]+"="+str(pnum), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
 cv2.imwrite(frame.split('.')[0] + '_result.jpg', img[...,::-1])
 cv2.destroyAllWindows()
 with open(frame.split('.')[0] + '_result.jpg', "rb") as f:
    # b64encode�����룬b64decode: ����
    base64_data = base64.b64encode(f.read())
    base=str(base64_data,'utf8') 
    # base64.b64decode(base64_data)
    context={
                   'h':hnum,
                   'p':pnum,
                   'data':base
}
    #����context��Ϣ�ؽӿ� 
    json_str = json.dumps(context)
    #return str(context)
    return json_str
if __name__ == "__main__":
    app.run("127.0.0.1", port=5000)