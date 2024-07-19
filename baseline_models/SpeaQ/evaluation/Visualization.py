
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
import h5py
import torch
import networkx as nx

def get_info_by_idx(inputs, outputs, thres=0.6):
    """
        Get the information of the detected objects and relations by the index
        The Groundtruth and prediction are stored in the input and output of the model

        Parameters:
            inputs (Dictionary): the stored the informatin of the input from the groundtruth
            outputs (Dictionary): the stored the informatin of the output from the prediction
            idx (int): the index of the image

        Returns:
            input_img(numpy.ndarray): The array of the input image
            boxes(list): The list of the detected box coordinates
            labels(list): The list of the ground truthlabels
            box_labels(list): The list of the detected box labels
            pred_scores(list): The list of the detected box scores
            pred_rels(list): The list of the detected relations
            gt_rels(list): The list of the ground truth relations
            pred_rel_score(list): The list of the detected relation scores
            pred_rel_label(list): The list of the detected relation labels
            output_img_size(list): The list of the output image size
    """
    #  groundtruth = det_input['groundtruths'][idx]
    #  prediction = det_input['predictions'][idx]
    # image path
    #   img_path = detected_info[idx]['img_file']
    input_img = inputs[0]['image']

    image_file = json.load(open('data/datasets/VG/image_data.json'))
    vocab_file = json.load(open('data/datasets/VG/VG-SGG-dicts-with-attri.json'))
    data_file = h5py.File('data/datasets/VG/VG-SGG-with-attri.h5', 'r')
    # boxes
    boxes = outputs[0]['instances'].get('pred_boxes').tensor.tolist()
   # boxes = inputs[0]['instances'].get('gt_boxes').tensor.tolist()
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx, idx2label[str(i+1)]) for idx, i in
              enumerate(inputs[0]['instances'].get('gt_classes').tolist())]
    pred_labels = ['{}-{}'.format(idx, idx2label[str(i+1)]) for idx, i in
                   enumerate(outputs[0]['instances'].get('pred_classes').tolist())]
    pred_scores = outputs[0]['instances'].get('scores')
    # get the top labels and scores and boxes
    pred_score_sorted, pred_score_idx = torch.sort(pred_scores, descending=True)
    pred_mask = torch.where(pred_score_sorted >  0.5)[0].tolist()
    pred_escending_mask = pred_score_idx[pred_mask].tolist()
    boxes = [boxes[i] for i in pred_escending_mask]
    box_labels = [pred_labels[i] for i in pred_escending_mask]
    #pred_scores = [pred_scores[i] for i in pred_escending_mask]

    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = inputs[0]['relations'].tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2]+1)], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = outputs[0]['rel_pair_idxs'].tolist()
    pred_rel_score = outputs[0]['pred_rel_scores']
    pred_rel_label = outputs[0]['pred_rel_labels']
    #pred_rel_label[:, 0] = 0
    pred_rel_score, pred_rel_ = pred_rel_score.max(-1)
    pred_rel_score_sorted, pred_rel_score_idx = torch.sort(pred_rel_score, descending=True)
    mask = torch.where(pred_rel_score_sorted > thres)[0].tolist()
    descending_mask = pred_rel_score_idx[mask].tolist()
   # pred_rel_score = pred_rel_score[mask]
    pred_rel_label = pred_rel_label.tolist()
    #only keep the top  relations and relation scores
    pred_rels = [(pred_labels[pred_rel_pair[i][0]], idx2pred[str(pred_rel_label[i]+1)], pred_labels[pred_rel_pair[i][1]]) for i in descending_mask ]
    pred_rel_score = pred_rel_score_sorted[mask]


    output_img_size =outputs[0]['instances'].image_size
    return input_img, boxes, labels, box_labels, pred_scores, pred_rels, gt_rels, pred_rel_score, pred_rel_label, output_img_size




def show_all(inputs, outputs, mode):
    """
        Visualize the detected objects and relations by the index

        Parameters:
            inputs (Dictionary): the stored the informatin of the input from the groundtruth
            outputs (Dictionary): the stored the informatin of the output from the prediction
            mode (String): the mode of the object transplanting

    """
    img_path, boxes, labels, box_labels, pred_scores, pred_rels, gt_rels, pred_rel_score, pred_rel_label, output_img_size = get_info_by_idx(
        inputs, outputs)

    draw_image(img=img_path, boxes=boxes, labels=labels, box_labels=box_labels, pred_scores=pred_scores,
               gt_rels=gt_rels, pred_rels=pred_rels, pred_rel_score=pred_rel_score, pred_rel_label=pred_rel_label, output_img_size=output_img_size,
               print_img=False, mode = mode)




def draw_single_box(img, box, color='red', draw_info=None, scale_x=1, scale_y=1):
    """
        Draw a single box on the image

        Parameters:
            img (numpy.ndarray): the image array
            box (List): the list of the box coordinates
            color (String): the color of the box
            draw_info (String): the information of the box
            scale_x (int): the scale of the x-axis
            scale_y (int): the scale of the y-axis

    """
    #   img = Image.fromarray(pic)
    draw = ImageDraw.Draw(img)
    bboxes_scaled = box#rescale_bboxes(box, img.size)
    #output image are saceld by 2, box need to be rescaled by multiplied by 2

    x1, y1, x2, y2 = int(bboxes_scaled[0]*scale_x), int(bboxes_scaled[1]*scale_y), int(bboxes_scaled[2]*scale_x), int(bboxes_scaled[3]*scale_y)
    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
    font = ImageFont.truetype("evaluation/Gidole-Regular.ttf", size=22)
    if draw_info:

        info = draw_info
        draw.text((x1, y1), info, font=font, fill = color)


def print_list(name, input_list, scores):
    """
        print the list of the input

        Parameters:
            name (String): the name of the list
            input_list (List): the list of the input
            scores (List): the list of the scores


    """
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i].item()))


def draw_image(img, boxes, labels, box_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label, output_img_size,
               print_img=False, mode = None):
    """
        Visualize the detected objects and relations by the index with the image

        Parameters:
            input_img(numpy.ndarray): The array of the input image
            boxes(list): The list of the detected box coordinates
            labels(list): The list of the ground truthlabels
            box_labels(list): The list of the detected box labels
            pred_scores(list): The list of the detected box scores
            pred_rels(list): The list of the detected relations
            gt_rels(list): The list of the ground truth relations
            pred_rel_score(list): The list of the detected relation scores
            pred_rel_label(list): The list of the detected relation labels
            output_img_size(list): The list of the output image size
            print_img(Boolean): the flag of the print image
            mode (String): the mode of the object transplant

    """
    pic = np.transpose(np.asarray(img), (1, 2, 0))
    pic_img = Image.fromarray(pic)
    acale_x = img.shape[1]/output_img_size[0]
    scale_y = img.shape[2]/output_img_size[1]

    num_obj = np.asarray(boxes).shape[0]
    for i in range(num_obj):
        info = box_labels[i] + ': ' + str("{:.3f}".format(pred_scores[i].item()))
        draw_single_box(pic_img, boxes[i], draw_info=info, scale_x=acale_x, scale_y=scale_y)
    #   if print_img:
    #       display(pic)
    fig, ax = plt.subplots()
    ax.imshow(pic_img)
    ax.axis('off')
    plt.savefig(mode +'.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels, None)
        print('*' * 50)
    print_list('gt_rels', gt_rels, None)
    print('*' * 50)
    print_list('pred_labels', box_labels, pred_scores)
    print('*' * 50)
    print_list('pred_rels', pred_rels, pred_rel_score)
    print('*' * 50)
    draw_relationship(pred_rels, pred_rel_score, mode)


    return None

def draw_relationship(pred_rels, pred_rel_score, mode):
    """
        Plot the detected relations

        Parameters:
            pred_rels(list): The list of the detected relations
            mode (String): the mode of the object transplant

    """

    # Open a file in write mode
    with open(mode + '_relationship.txt', 'w') as file:
        for i, rel in enumerate(pred_rels):
            file.write(f"{'pred_rels ' + str(i) + ': ' + str(rel) + '; score: ' + str(pred_rel_score[i].item())}\n")
    # Create a directed graph
    G = nx.Graph()

    # Add edges to the graph based on the top 5 relations
    for i in range(5):
        subject, predicate, obj = pred_rels[i]
        prob = pred_rel_score[i].item()
        G.add_edge(subject, obj, label=predicate, weight=prob)


    # Set up the plot
    plt.figure(figsize=(8, 6))

    # Draw the graph with labels
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    nx.draw(G, pos, with_labels=True, node_size=30, node_color='skyblue', font_size=5, font_weight='bold',
            arrows=True)

    # Draw edge labels with predicates and probabilities
    edge_labels = {(u, v): f"{d['label']} ({d['weight']:.2f})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',font_size=5)

    # Save the plot as a PDF with small margins
    plt.title("Relation Graph with Probabilities")
    plt.savefig(mode +'_relation_graph.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()