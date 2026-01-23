import argparse
import csv
import cv2

import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
import os
import ast
import numpy as np
import re
from shapely.geometry import Polygon
from shapely.geometry import mapping

def create_object(name, xmin, ymin, xmax, ymax):
    obj = ET.Element('object')
    
    name_element = ET.SubElement(obj, 'name')
    name_element.text = name

    pose_element = ET.SubElement(obj, 'pose')
    pose_element.text = 'Unspecified'
    
    truncated_element = ET.SubElement(obj, 'truncated')
    truncated_element.text = str(0)
    
    difficult_element = ET.SubElement(obj, 'difficult')
    difficult_element.text = str(0)

    bndbox = ET.SubElement(obj, 'bndbox')
    xmin_element = ET.SubElement(bndbox, 'xmin')
    xmin_element.text = str(xmin)
    
    ymin_element = ET.SubElement(bndbox, 'ymin')
    ymin_element.text = str(ymin)
    
    xmax_element = ET.SubElement(bndbox, 'xmax')
    xmax_element.text = str(xmax)
    
    ymax_element = ET.SubElement(bndbox, 'ymax')
    ymax_element.text = str(ymax)
    
    return obj

def parse_image_path_2xml(image_path):
    
    current_video = Path(image_path).parent.name
    current_parent = Path(image_path).parent.parent.name
    current_video = current_parent+'/'+current_video
    image_number =  Path(image_path).stem.split(".")[0]
   
    return current_video, image_number

def binary_mask_to_polygon(binary_mask):
    tolerance = 1
    binary_mask = (binary_mask > 0).astype(np.uint8)
    #print(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    if not contours :
        #print('inzo....')
        return None,None
    polygons = []
    contour = max(contours, key=len)
    
    #points = [f"{point[0][0]},{point[0][1]}" for point in contour]
    contour_tuple = [tuple(point[0]) for point in contour]
    #print('length::::',len(contour_tuple))
    if len(contour_tuple) < 20:
        return None,None
    area = cv2.contourArea(contour)
    
    polygon = Polygon(contour_tuple)
    simplified_polygon = polygon.simplify(tolerance, preserve_topology=True)
    simplified_coords = list(mapping(simplified_polygon)["coordinates"])
    simplified_data = list(simplified_coords[0])
    if len(simplified_data) > 20:
        #print(simplified_data)
        polygon_string = ";".join([f"{int(x)},{int(y)}" for x, y in simplified_data])
        #polygon_string = ";".join(simplified_data) + ";"  
        #polygons.append(polygon_string)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if area < 5000 or aspect_ratio > 5 or aspect_ratio < 0.2:
            return None,None
        bbox = (x, y, x + w, y + h) 
        return polygon_string,bbox
    else:
        return None,None
    #print('++',len(simplified_data))

def polygon_to_string(polygon):
    if isinstance(polygon, list):
        polygon = np.array(polygon)
    return ';'.join([f"{int(x)},{int(y)}" for x, y in polygon])

def polygon_to_bbox(polygon):
    #print('######',polygon)
    #polygon = polygon
    #print(polygon)
    if len(polygon) < 10:
        return None, None 
    polygon_str = polygon_to_string(polygon)
    
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    x_min = int(min(xs))
    y_min = int(min(ys))
    x_max = int(max(xs))
    y_max = int(max(ys))
    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    #if area < 2000:
    #    return None, None
    
    return [x_min, y_min, x_max, y_max],polygon_str

def binary_mask_to_bbox_numpy(binary_mask):
    
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    rows, cols = np.where(binary_mask > 0)
    
    if rows.size == 0 or cols.size == 0:
       
        return None
    
    x1, y1 = cols.min(), rows.min()
    x2, y2 = cols.max(), rows.max()
    
    return (x1, y1, x2, y2)

def write_to_xml(image_file,info, save_folder, database_name):
    
    objects = ''
    image_folder_name ,image_name = parse_image_path_2xml(image_file)
    #image_folder_name =image_folder_name.split('/')[-1]
    #print('image_file:::',image_file)
    with Image.open(image_file) as img:
        width, height = img.size
        if img.mode == 'YCbCr':
            depth = 3
        else:
            depth = len(img.mode)
    
    xml_file = os.path.join(save_folder,image_folder_name,(image_name+'.xml'))
    #print('xml_',xml_file)
    tolerance = 1
    if not os.path.isfile(xml_file):
        
        for i, data in enumerate(info):
            #bbox = data['box']
            polygon_mask,bbox = binary_mask_to_polygon(data['mask'])
            if polygon_mask==None or bbox == None:
                continue
            objects = objects + '''
            <object>
                <name>{category_name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{xmin}</xmin>
                    <ymin>{ymin}</ymin>
                    <xmax>{xmax}</xmax>
                    <ymax>{ymax}</ymax>
                </bndbox>
                <mask>{mask}</mask>
            </object>'''.format(
                    category_name = i,
                    xmin = bbox[0],
                    ymin = bbox[1],
                    xmax = bbox[2],
                    ymax = bbox[3],
                    mask = polygon_mask,
                )
           
        if not objects:
            return
        
        xml = '''<annotation>
	    <folder>{image_folder_name}</folder>
	    <filename>{image_name}</filename>
	    <source>
		    <database>{database_name}</database>
	    </source>
	    <size>
		    <width>{width}</width>
		    <height>{height}</height>
	    </size>
	    <segmented>0</segmented>{objects}
        </annotation>'''.format(
        image_folder_name = image_folder_name,
        image_name = image_name,
        database_name = database_name,
        width = width,
        height = height,
        objects = objects
    )
       
        folder = os.path.join(save_folder,image_folder_name)
        #print('xml_file::',xml_file)
        
        if not os.path.exists(folder):
            os.mkdir(folder)
         
        with open(xml_file, 'w') as file:
            file.write(xml)
        
    '''
    else:
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for bbox in bboxes:
            new_object = create_object(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])
            root.append(new_object)
        
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
    '''
def write_to_xml_box(image_file,bboxes, save_folder, database_name):
    
    objects = ''
    image_folder_name ,image_name = parse_image_path_2xml(image_file)
    #image_folder_name =image_folder_name.split('/')[-1]
    #print('image_file:::',image_file)
    with Image.open(image_file) as img:
        width, height = img.size
        if img.mode == 'YCbCr':
            depth = 3
        else:
            depth = len(img.mode)
    
    xml_file = os.path.join(save_folder,image_folder_name,(image_name+'.xml'))
    #print('xml_',xml_file)
    tolerance = 1
    if not os.path.isfile(xml_file):
        for i, bbox in enumerate(bboxes):
            
            objects = objects + '''
            <object>
                <name>{category_name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{xmin}</xmin>
                    <ymin>{ymin}</ymin>
                    <xmax>{xmax}</xmax>
                    <ymax>{ymax}</ymax>
                </bndbox>
            </object>'''.format(
                    category_name = i,
                    xmin = bbox[0],
                    ymin = bbox[1],
                    xmax = bbox[2],
                    ymax = bbox[3],
                )
            
        if not objects:
            return
        
        xml = '''<annotation>
	    <folder>{image_folder_name}</folder>
	    <filename>{image_name}</filename>
	    <source>
		    <database>{database_name}</database>
	    </source>
	    <size>
		    <width>{width}</width>
		    <height>{height}</height>
	    </size>
	    <segmented>0</segmented>{objects}
        </annotation>'''.format(
        image_folder_name = image_folder_name,
        image_name = image_name,
        database_name = database_name,
        width = width,
        height = height,
        objects = objects
        )
             
        folder = os.path.join(save_folder,image_folder_name)
        
        print('xml_file::',xml_file)
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        with open(xml_file, 'w') as file:
            file.write(xml)


def write_to_xml_SAM(image_file,info, save_folder, database_name,confidence_score=0.87):
    
    objects = ''
    image_folder_name ,image_name = parse_image_path_2xml(image_file)
    #image_folder_name =image_folder_name.split('/')[-1]
    #print('image_file:::',image_file)
    with Image.open(image_file) as img:
        width, height = img.size
        if img.mode == 'YCbCr':
            depth = 3
        else:
            depth = len(img.mode)
    
    xml_file = os.path.join(save_folder,image_folder_name,(image_name+'.xml'))
    #print(xml_file)
    tolerance = 1
    if not os.path.isfile(xml_file):
        for i, data in enumerate(info):
            
            confidence = data['confidence_score']
            if float(confidence) >confidence_score:
                bbox = data['box']
                polygon_mask = data['mask']
                if polygon_mask == None:
                    continue
                
                objects = objects + '''
            <object>
                <name>{category_name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{xmin}</xmin>
                    <ymin>{ymin}</ymin>
                    <xmax>{xmax}</xmax>
                    <ymax>{ymax}</ymax>
                </bndbox>
                <mask>{mask}</mask>
                <confidence>{confidence}</confidence>
            </object>'''.format(
                    category_name = i,
                    xmin = bbox[0],
                    ymin = bbox[1],
                    xmax = bbox[2],
                    ymax = bbox[3],
                    mask = polygon_mask,
                    confidence = confidence
                )
        if not objects:
            return
        xml = '''<annotation>
	    <folder>{image_folder_name}</folder>
	    <filename>{image_name}</filename>
	    <source>
		    <database>{database_name}</database>
	    </source>
	    <size>
		    <width>{width}</width>
		    <height>{height}</height>
	    </size>
	    <segmented>0</segmented>{objects}
        </annotation>'''.format(
        image_folder_name = image_folder_name,
        image_name = image_name,
        database_name = database_name,
        width = width,
        height = height,
        objects = objects
        )
        

        folder = os.path.join(save_folder,image_folder_name)
        
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        #print('xml...',xml_file)
        with open(xml_file, 'w') as file:
            file.write(xml)

def write_to_xml_vit(image_file,info, save_folder, database_name):
    
    objects = ''
    image_folder_name ,image_name = parse_image_path_2xml(image_file)
    #image_folder_name =image_folder_name.split('/')[-1]
    #print('image_file:::',image_file)
    with Image.open(image_file) as img:
        width, height = img.size
        if img.mode == 'YCbCr':
            depth = 3
        else:
            depth = len(img.mode)
    
    xml_file = os.path.join(save_folder,image_folder_name,(image_name+'.xml'))
    #print('xml_',xml_file)
    tolerance = 1
    if not os.path.isfile(xml_file):
        #print('inototototo')
        for i, poly_mask in enumerate(info):
            #bbox = data['box']
            #print('yyy')
            bbox,mask_str=polygon_to_bbox(poly_mask)
            #print('bbox',bbox)
            
            if len(poly_mask)==0 or bbox == None:
                continue
            #print('go')
            objects = objects + '''
            <object>
                <name>{category_name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{xmin}</xmin>
                    <ymin>{ymin}</ymin>
                    <xmax>{xmax}</xmax>
                    <ymax>{ymax}</ymax>
                </bndbox>
                <mask>{mask}</mask>
            </object>'''.format(
                    category_name = i,
                    xmin = bbox[0],
                    ymin = bbox[1],
                    xmax = bbox[2],
                    ymax = bbox[3],
                    mask = mask_str,
                )
            
        if not objects:
            return
        
        xml = '''<annotation>
	    <folder>{image_folder_name}</folder>
	    <filename>{image_name}</filename>
	    <source>
		    <database>{database_name}</database>
	    </source>
	    <size>
		    <width>{width}</width>
		    <height>{height}</height>
	    </size>
	    <segmented>0</segmented>{objects}
        </annotation>'''.format(
        image_folder_name = image_folder_name,
        image_name = image_name,
        database_name = database_name,
        width = width,
        height = height,
        objects = objects
        )
        #print('folder',save_folder)
        folder = os.path.join(save_folder,image_folder_name)
        
        
        if not os.path.exists(folder):
            #print('into..')
            os.mkdir(folder)
        print('xml',xml_file)
        with open(xml_file, 'w') as file:
            file.write(xml)

def write_to_xml_predictor(image_file,info, save_folder, database_name):
    
    objects = ''
    image_folder_name ,image_name = parse_image_path_2xml(image_file)
    #image_folder_name =image_folder_name.split('/')[-1]
    #print('image_file:::',image_file)
    with Image.open(image_file) as img:
        width, height = img.size
        if img.mode == 'YCbCr':
            depth = 3
        else:
            depth = len(img.mode)
    
    xml_file = os.path.join(save_folder,image_folder_name,(image_name+'.xml'))
    
    tolerance = 1
    if not os.path.isfile(xml_file):
        for i, data in enumerate(info):
            #bbox = data['box']
            
            bbox = data['box']
            polygon_mask = data['mask']
            if polygon_mask == None:
                continue
                
            #polygon_mask = re.sub(r"^\['|'\]$", "", str(polygon_mask))
            #print('mask......',polygon_mask)
            #polygon_mask = data['mask']
            #bbox = binary_mask_to_bbox_numpy(polygon_mask)
            objects = objects + '''
	    <object>
		    <name>{category_name}</name>
		    <pose>Unspecified</pose>
		    <truncated>0</truncated>
		    <difficult>0</difficult>
		    <bndbox>
			    <xmin>{xmin}</xmin>
			    <ymin>{ymin}</ymin>
			    <xmax>{xmax}</xmax>
			    <ymax>{ymax}</ymax>
		    </bndbox>
            <mask>{mask}</mask>
	    </object>'''.format(
                category_name = i,
                xmin = bbox[0],
                ymin = bbox[1],
                xmax = bbox[2],
                ymax = bbox[3],
                mask = polygon_mask
            )
        if not objects:
            return
        xml = '''<annotation>
	    <folder>{image_folder_name}</folder>
	    <filename>{image_name}</filename>
	    <source>
		    <database>{database_name}</database>
	    </source>
	    <size>
		    <width>{width}</width>
		    <height>{height}</height>
	    </size>
	    <segmented>0</segmented>{objects}
        </annotation>'''.format(
        image_folder_name = image_folder_name,
        image_name = image_name,
        database_name = database_name,
        width = width,
        height = height,
        objects = objects
    )
        folder = os.path.join(save_folder,image_folder_name)
        print('xml_file::',xml_file)
        
        if not os.path.exists(folder):
            os.mkdir(folder)
         
        with open(xml_file, 'w') as file:
            file.write(xml)

def main():
    parser = argparse.ArgumentParser(description='Coco Json to Pascal VOC XML Converter.')
    parser.add_argument('--csv_file',default='/netscratch/zlu/dataset/epic-kitchen/annotations/EPIC_test_seen_object_labels.csv' , help='Target json file to convert to csv.')
    parser.add_argument('--image_folder', default='/netscratch/zlu/dataset/epic-kitchen/images/', help='Target folder to find images.')
    parser.add_argument('--save_xml',default= '/netscratch/zlu/dataset/epic-kitchen/annotations/seenval_xml', help='The folder to save annotations xmls.')
#parser.add_argument('--database_name', required=False, default='', help='The name of database.')
    parser.add_argument('--no_skip_background', dest='skip_background', action='store_false', help='Do not skip \'background\' category.')
    parser.set_defaults(skip_background=True)
    args = parser.parse_args()
    
    Path(args.save_xml).mkdir(parents=True, exist_ok=True)
    image_folder = '/netscratch/zlu/dataset/epic-kitchen/images/'
    with open('/netscratch/zlu/dataset/epic-kitchen/annotations/EPIC_test_seen_object_labels.csv') as csvfile:

        dict_reader = csv.DictReader(csvfile)
    #print('Write annotations file...')
        now = 1
        for row in dict_reader:
            image_folderpath = os.path.join(row['participant_id'],'object_detection_images',row['video_id'])
            
            for img in os.listdir(os.path.join(image_folder, row['participant_id'],'object_detection_images',row['video_id'])):
                if row['frame'] in img:
                    imgfile = os.path.join(image_folderpath,img)
                    #print('imgfile',imgfile)
            if '{' in row['bounding_boxes']:
                new = []
                bounding_boxes = row['bounding_boxes']
                if bounding_boxes.count('{') == 1:
                    new.append(bounding_boxes)
                elif bounding_boxes.count('{')>1:
                    new=bounding_boxes.split('}')
                    new.remove('')
                    for i in range(0,len(new)):
                        new[i]=new[i]+'}'
                anno_list =[] 
                for j in range(0,len(new)):
                    new[j] =ast.literal_eval(new[j])
                    anno_list.append([row['noun'], int(new[j]['left']), int(new[j]['top']), (int(new[j]['left'])+int(new[j]['width'])), (int(new[j]['top'])+int(new[j]['height']))])
                #print(anno_list)
            database_name = 'epic-kitchen'
        #print(filename)
            write_to_xml(imgfile, anno_list, image_folderpath,image_folder, args.save_xml,database_name)
            #print('Write xml files ({} / {})'.format(now, total))
            now = now + 1
            print('Write xml files ({} / {})'.format(now, imgfile))
        
    #print('Annotations file was written!')



if __name__ == '__main__':
    
    main()
   
