# import json
import xml.etree.ElementTree as ET

"""
无论是xml还是json,其实就是一种包含了数据以及数据说明的文本格式规范，数据时一样的，不同只是数据的格式(包装数据的不同格式而已)
"""

"""
=================================================================================================
XML解析
XML被设计用来传输和存储数据
python解析XML的三种方法，SAX，DOM，ElementTree

XML-ElementTree
<tag attrib>
    pass #text or data
</tag>

import xml.etree.ElementTree as ET
#读取
tree=ET.parse('certain_doc.xml')
或 tree=ET.ElementTree(file='certain_xml')
#根节点
root=tree.getroot() #root是可迭代的，root.findall()也是可迭代的，element中的text可以用索引表示
#属性tag、text、attrib、tail
root.tag
root.text
root.attrib

element.findall(tag).text #返回的是所有匹配的tag的列表
element.find(tag).text #返回第一个匹配的tag的元素
element.iter(tag) #以当前元素为根节点，创建迭代器，且以tag进行过滤

遍历：
1、简单遍历(全遍历)
for child in root:
    pass
2、直接访问节点
certain_text=root[i][...j].text
3、
for object in root.iter(certain_tag):
    pass
4、    
for object in root.findall(certain_tag) 

"""

"""
=================================================================================================
json

json是一种轻量级的数据交换格式，
python 可以使用json模块来对JSON数据进行编解码
json.dumps() 对数据进行编码
json.loads() 对数据进行编码
python->json
dict->object
list/tuple->array
str->string
int...->number
true/false->true/false
none->null

json_objet=json.dumps(python_object)
python_object2=json.loads(json_object2)
with open('file.json','w') as f:
    json.dump(data,f)
with open('file.json','r') as f:
    data=json.loads(f)
"""



def create_label_map():
    #lable map
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    #将标签变成整数，值得注意的是label_map是字典的形式
    label_map={k:v+1 for v,k in enumerate(voc_labels)}
    label_map['background']=0

    return voc_labels,label_map #返回原始标签和label字典


def parse_annotations(annotation_path):

    #得到lable_map
    _,label_map=create_label_map()

    #解析xml文件
    tree=ET.parse(annotation_path)
    #获得根节点
    root=tree.getroot()

    boxes=list() #bbox列表
    labels=list() #标签列表
    difficulties=list() #难识别列表

    #对一张图像(训练集和验证集中)解析真值ground-truth列表
    #选出所有符合object-tag的元素，创建迭代器
    for object in root.iter('object'):
        difficult=int(object.find('difficult').text=='1') #将逻辑值True和False转换为0，1

        label=object.find('name').text.lower().strip() #str.lower()将所有大写转换为小写，str.strip()移除字符串头尾指定的字符(默认为空格)
        if label not in label_map:
            continue

        #查找Bbox
        #bbox定义的是左上和右下的坐标
        #...???...这里为什么要减1，还有数值的问题，[0,1]
        bbox=object.find('bndbox')
        xmin=int(bbox.find('xmin').text)-1
        ymin=int(bbox.find('ymin').text)-1
        xmax=int(bbox.find('xmax').text)-1
        ymax=int(bbox.find('ymax').text)-1

        #创建列表
        boxes.append([xmin,ymin,xmax,ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    #以字典的方式返回解析到的ground-truth
    return {'boxes':boxes,'labels':labels,'difficulties':difficulties}


"""
一个模块被另一个程序第一次引入的时候，其主程序将被执行。如果想在模块被引入时，模块中的某一程序块不执行，就可以用__name__属性来是该程序块仅在该模块自身运行时执行
每个模块都有一个__name__属性，当其值为__main__时，表明该模块自身在运行，否则是被引入
"""
if __name__=='__main__':
    create_label_map()
    parse_annotations()
