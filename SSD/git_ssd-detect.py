import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image,ImageFont,ImageDraw

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载模型checkpoint
#其实就是正经保存的模型
checkpoint='./SSD/checkpoint_ssd300.pth.tar'
checkpoint=torch.load(checkpoint)
#相当于
# class pass
# training_loop pass
# 已经训练好的net
# torch.save(net,'net.pkl')
# net2=torch.load('net.pkl')

start_epoch=checkpoint['epoch']+1
print('\nLoaded checkpoint from epoch %d.\n'% start_epoch)
model=checkpoint['model']
model=model.to(device)
model.eval()

#数据转换
resize=transforms.Resize((300,300))
to_tensor=transforms.ToTensor()
normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

def detect(original_image,min_score,max_overlop,top_k,suppress=None):
    """
    利用SSD300模型进行检测，并可视化结果
    :param original_image: a PIL image
    :param min_score: 能够与特定类别匹配的检测框的阈值
    :param max_overlop: 不执行NMS的最大阈值
    :param top_k: 保留前k个类
    :param suppress: 确定不是或不想存在于图像的类，list
    :return: 标注后的图像
    """

    #转换
    image=normalize(to_tensor(resize(original_image)))

    #移动到默认的处理设备上
    predicted_locs,predicted_scores=model(image.unsqueeze(0))

    #在SSD的输出中检测目标
    det_boxes,det_labels,det_scores=model.detect_objects(predicted_locs,predicted_scores,min_score=min_score,max_overlap=max_overlop,top_k=top_k)

    #将检测移动到cup
    det_boxes=det_boxes[0].to('cpu')

    #转换成原始的维度
    original_dims=torch.FloatTensor([original_image.width,original_image.height,original_image.width,original_image.height]).unsqueeze(0)
    det_boxes=det_boxes*original_dims #本来是小数形式，现在变成了绝对坐标形式

    #类别解码
    det_labels=[rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels==['background']:
        #直接返回原始图像
        return original_image

    #注释
    annotated_image=original_image
    draw=ImageDraw.Draw(annotated_image)
    font=ImageFont.truetype("./calibril.ttf",15)

    #抑制特定的类别，如果需要的话
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        #boxes
        box_location=det_boxes[i].tolist()
        draw.rectangle(xy=box_location,outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l+1. for l in box_location],outline=label_color_map[det_labels[i]])

        #text
        text_size=font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',font=font)

    del draw

    return annotated_image

if __name__=='__main__':
    img_path=''
    original_image=Image.open(img_path,mode='r')
    original_image=original_image.convert('RGB')
    detect(original_image,min_score=0.2,max_overlop=0.5,top_k=200).show()


