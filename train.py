import torch
import time
from tqdm import tqdm
import numpy as np
from model_new import SegNet, DenseASPP
import torch.nn as nn
from IPython.display import clear_output
from scipy.special import softmax
import matplotlib.pyplot as plt
import cv2
import sys
import os


ignore_index = 0
train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']

class_map = dict(zip(train_id, range(len(train_id))))

def get_uoi(one, two):
    intersection = np.logical_and(one, two)
    union = np.logical_or(one, two)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

colors = [[  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(train_id, colors))

def create_mask(pred_mask):
    pred_mask = np.argmax(softmax(pred_mask), axis=1)
    pred_mask = pred_mask[...,np.newaxis]

    return decode_segmap(pred_mask)

def show_predictions(model, dataset=None, num=1, epoch=-1, sample_image=None, sample_mask=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model
    if dataset:
        for image, mask in dataset:
            image, mask = image[0], mask[0]
            image, mask = image.to(device), mask.to(device)
            pred_mask = model(image[np.newaxis, ...]).cpu().detach().numpy()
            pred = create_mask(pred_mask)
            display([image.cpu().detach().numpy(), decode_segmap(mask[np.newaxis,...].cpu().detach().numpy()), pred])
            break
            
    else:
        image = sample_image[np.newaxis, ...].to(device)
        pred = create_mask(model(image).cpu().detach().numpy())
        display([sample_image, decode_segmap(sample_mask.cpu().detach().numpy()), pred])
    print('UOI : ', get_uoi(decode_segmap(sample_mask.cpu().detach().numpy()), pred))
    pred = np.reshape(pred*255, (256, 256, 3)).astype(np.int16)
    cv2.imwrite(f'pred_torch/paper_{epoch+1}.png', pred)

def decode_segmap(temp):
    #convert gray scale to color
    shape = (256, 256)
    temp = np.squeeze(temp)
    #temp = temp[0,:,:]
    print(temp.shape)
    r = np.zeros((shape))
    g = np.zeros((shape))
    b = np.zeros((shape))
    for i in train_id:
        r[temp == i] = label_colours[i][0]
        g[temp == i] = label_colours[i][1]
        b[temp == i] = label_colours[i][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    
    try: 
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
    except:
        rgb[:, :, 0] = r.reshape(1, -1) / 255.0
        rgb[:, :, 1] = g.reshape(1, -1) / 255.0
        rgb[:, :, 2] = b.reshape(1, -1) / 255.0
        
    return rgb

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = display_list[i]
        img = np.reshape(img, (256, 256, 3))
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    
def train(train_loader, val_loader=None, EPOCHS=10, name='base', continue_epoch=0, state_dict=None):
    inference_time = []
    
    #model = DenseASPP(in_channels=3, num_classes=19)
    model = SegNet()
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    model.to(device)
        
    for x, y in train_loader:
        sample_image, sample_mask = x[0], y[0]
        break
        
    for s in range(EPOCHS):
        running_loss = 0.0
        
        model.train()
        iterator = tqdm(train_loader)
        for i, data in enumerate(iterator, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            pred_class = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_time.append(time.time()-start_time)
            pred_class = pred_class.to(torch.float)
            labels = labels.to(torch.long)
            loss = loss_fn(pred_class, labels)
            del inputs, labels, pred_class
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iterator.set_description('Epoch: {}, loss: {:.4f}'.format(s+1, running_loss/(i+1)))
               
        #scheduler.step()
        
        with torch.no_grad():
            if val_loader:
                pred_label = []
                real_labels = []
                loss_all = []
                iterator = tqdm(val_loader)
                for i, data in enumerate(iterator, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.to(torch.long)
                    pred_class = model(inputs)
                    pred_class = pred_class.to(torch.float)
                    pred_label.append(pred_class.cpu().detach().numpy())
                    real_labels.append(labels.cpu().detach().numpy())
                    loss = loss_fn(pred_class, labels)
                    loss_all.append(loss.cpu().detach().numpy())
                    
                    del inputs, labels, pred_class
                    torch.cuda.empty_cache()

                pred_label = np.cat(pred_label, axis=0).to(torch.float32)
                real_labels = np.cat(real_labels, axis=0).to(torch.float32)
                label_acc = (np.argmax(softmax(pred_label, dim=1), dim=1).round() == real_labels).float().mean()
                print('Label accuracy: {:.4f}, Loss {:.4f}'.format(label_acc, loss))

            else:
                label_acc = 0
                loss_all = 0

        clear_output(wait=True)
        show_predictions(model, epoch=s, sample_image=sample_image, sample_mask=sample_mask)
        print ('\nSample Prediction after epoch {}\n'.format(s+1+continue_epoch))
                
        # save the model
        file = 'models/' + name + '_' + f'{s+continue_epoch}'
        torch.save({
                    'epoch': s,
                    'train_loss': running_loss,
                    'val_loss': np.mean(loss_all),
                    'label_acc': label_acc,
                    'weights': model.state_dict()
                    }, file)
        
    print('Average Inference Time:', np.sum(inference_time)/EPOCHS)
    print('Std of Infernce Time:', np.std(inference_time))
    
    
def get_uoi(one, two):
    intersection = np.logical_and(one, two)
    union = np.logical_or(one, two)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_dice(one, two):
    intersection = np.sum(np.logical_and(one, two))*2
    area = (256*256*3)*2
    dice = intersection/area
    return dice
    
def test(val_loader, weights, name='base', continue_epoch=0):
    
    for x, y in val_loader:
        sample_image, sample_mask = x[0], y[0]
        break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for s in range(0, len(weights)):
            acc = []
            loss = []
            model = SegNet()
            state_dict = torch.load(os.path.join('models/', weights[s]))
            model.load_state_dict(state_dict['weights'])
            model.to(device)
            iterator = tqdm(val_loader)
            for i, data in enumerate(iterator, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                pred_label = model(inputs)
                pred_label = pred_label.to(torch.float)
                labels = labels.to(torch.long)
                loss.append(loss_fn(pred_label, labels).cpu().detach().numpy())
                pred_label = torch.argmax(torch.nn.functional.softmax(pred_label, dim=1), dim=1)
                acc.append(get_uoi(labels.cpu().detach().numpy(), pred_label.cpu().detach().numpy()))
                

            file = weights[s]
            torch.save({'name' : weights[s],
                        'epoch': state_dict['epoch'],
                        'train_loss':  state_dict['train_loss'],
                        'val_loss': np.mean(loss),
                        'label_acc': np.mean(acc),
                        'weights': state_dict['weights']
                        }, file)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            model.to(device)
        

        print ('\nSample Prediction after epoch {}\n'.format(s+1+continue_epoch))
         # save the model

            
def get_test_pics(val_loader, weights, name='base', continue_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    for x, y in val_loader:
        sample_image, sample_mask = x[0], y[0]
        break            
    with torch.no_grad():
        for s in range(0, len(weights)):
            print(weights[s])
            model = SegNet()
            state_dict = torch.load(weights[s])
            model.load_state_dict(state_dict['weights'])
            model.to(device)
            iterator = tqdm(val_loader)
            #pred_img = model(sample_image)
            #pred_img = torch.argmax(torch.nn.functional.softmax(pred_label, dim=1), dim=1)
            show_predictions(model, epoch=s, sample_image=sample_image, sample_mask=sample_mask)
            
def get_y(val_loader, weights):
     
    model = SegNet()
    model.load_state_dict(weights[s])
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
             
    y_true = []
    y_pred = []
    for i, data in enumerate(iterator, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    y_true.append(labels.cpu().detach().numpy().flatten())
                    y_pred.apend(model(inputs).cpu().detach().numpy().flatten())
    return y_true, y_pred