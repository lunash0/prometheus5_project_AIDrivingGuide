import xml.etree.ElementTree as ET
import json
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_xml_for_person(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []
    annotations = []
    image_id_mapping = {}
    annotation_id = 0

    for image in root.findall('image'):
        image_id = int(image.get('id'))
        file_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        image_id_mapping[image_id] = len(images) - 1  # Map the original image ID to its index

        for box in image.findall('box'):
            label = box.get('label')

            if label in ["person", "dog", "cat"]:
                category_id = 0 if label == "person" else 1

                occluded = int(box.get('occluded')) if box.get('occluded') is not None else 0
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                bbox_width = xbr - xtl
                bbox_height = ybr - ytl

                z_order = int(box.get('z_order')) if box.get('z_order') is not None else 0

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,  # Use the original image ID
                    "category_id": category_id,
                    "bbox": [xtl, ytl, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "occluded": occluded,
                    "z_order": z_order
                })
                annotation_id += 1

    return images, annotations

def process_and_save(xml_files, output_file_path):
    cumulative_images = []
    cumulative_annotations = []
    cumulative_categories = [{"id": 0, "name": "person"}, {"id": 1, "name": "objects"}]

    image_id_counter = 0
    annotation_id_counter = 0

    for xml_file_path in tqdm(xml_files):
        images, annotations = parse_xml_for_person(xml_file_path)

        # Map the image ID in the current XML file to a new ID
        current_image_id_map = {img['id']: image_id_counter + i for i, img in enumerate(images)}
        
        for img in images:
            img['id'] = image_id_counter
            cumulative_images.append(img)
            image_id_counter += 1

        for ann in annotations:
            ann['id'] = annotation_id_counter
            ann['image_id'] = current_image_id_map.get(ann['image_id'], ann['image_id'])  # Remap image ID
            cumulative_annotations.append(ann)
            annotation_id_counter += 1

    coco_format_data = {
        "images": cumulative_images,
        "annotations": cumulative_annotations,
        "categories": cumulative_categories
    }

    with open(output_file_path, 'w') as json_file:
        json.dump(coco_format_data, json_file, indent=4)

    print(f'Combined annotations saved to {output_file_path}')

def main():
    dataset_base_folder = '/data/tmp/data'
    xml_files = []
    for subdir, _, files in os.walk(dataset_base_folder):
        for file in files:
            if file.endswith('.xml'):
                xml_file_path = os.path.join(subdir, file)
                xml_files.append(xml_file_path)
    xml_files = sorted(xml_files)
    train_files, val_files = train_test_split(xml_files, test_size=0.3, random_state=42)

    train_output_path = '/data/tmp/train_annotations.json'
    process_and_save(train_files, train_output_path)

    val_output_path = '/data/tmp/val_annotations.json'
    process_and_save(val_files, val_output_path)

if __name__ == "__main__":
    main()
