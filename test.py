from pathlib import Path
import cv2
import torch
import numpy as np
import os
from util.yolopv2_utils import letterbox
from util.yolopv2_utils import (
    select_device, increment_path,
    non_max_suppression, split_for_trace_model,
    driving_area_mask, show_seg_result, lane_line_mask,
    LoadImages
)
from ultralytics import YOLO
import models
from options.test_options import TestOptions  # Assuming this is part of your project

def detect():

    opt = TestOptions().parse()
    # Setting and directories
    source, seg_weights, save_txt, imgsz, output_folder = opt.source, opt.segweights, opt.save_txt, opt.img_size, opt.output_folder

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    segmentation_model = torch.jit.load(seg_weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    segmentation_model = segmentation_model.to(device)
    if half:
        segmentation_model.half()  # to FP16  
    segmentation_model.eval()
    # Load YOLO model for object detection
    detection_model = YOLO(opt.detection_weights)
    
    # Load inpainting model using the options from TestOptions
    inpaint_model = models.create_model(opt)
    inpaint_model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=32)

    # Run inference
    for counter,(path, img, im0s, vid_cap) in enumerate(dataset):
        print(im0s.shape)
        
        # Step 1: Object Detection using YOLO and Binary Mask Generation
        results = detection_model(im0s)
        height, width = im0s.shape[0], im0s.shape[1]
        binary_image = np.zeros((height, width), dtype=np.uint8)
        
        for result in results[0]:
            box = result.boxes.xyxy.cpu().numpy()[0]  # Bounding boxes
            x1, y1, x2, y2 = map(int, box)
            binary_image[y1:y2, x1:x2] = 1
        
        # Optionally dilate the binary mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(binary_image, kernel, iterations=10)

        # Step 2: Image Inpainting using the Generated Binary Mask
        img_resized = cv2.resize(im0s, (256, 256))  # Resize image
        dilated_mask_resized = cv2.resize(dilated_mask, (256, 256))  # Resize mask
        _, dilated_mask_resized = cv2.threshold(dilated_mask_resized, 0.5, 1, cv2.THRESH_BINARY)
        img_resized = img_resized.astype(np.float32) / 255.0 * 2 - 1  # Normalize image to [-1, 1]

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(dilated_mask_resized).unsqueeze(0).unsqueeze(0).float()

        # Create the data_i dictionary for the inpainting model
        data_i = {'image': img_tensor, 'mask': mask_tensor, 'path': ["output_image_path.jpg"]}

        # Perform inpainting
        with torch.no_grad():
            generated, _ = inpaint_model(data_i, mode='inference')


            # Post-process the generated image
            generated = torch.clamp(generated, -1, 1)
            generated = (generated + 1) / 2 * 255
            generated = generated.cpu().numpy().astype(np.uint8)
            pred_im = generated[0].transpose((1, 2, 0))

        

        # Step 3: Use the inpainted image as input to YOLO for further processing
        pred_im_resized = cv2.resize(pred_im[:, :, ::-1], (1280, 720))  # Resize inpainted image to YOLO input size
        img_yolop_input = letterbox(pred_im_resized, imgsz, stride=32)[0]
        
        # Convert
        img_yolop_input = img_yolop_input[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_yolop_input = np.ascontiguousarray(img_yolop_input)
        
        # Run YOLO inference (for lane detection, etc.)
        img_yolop_input = torch.from_numpy(img_yolop_input).to(device)
        img_yolop_input = img_yolop_input.half() if half else img_yolop_input.float()  # uint8 to fp16/32
        img_yolop_input /= 255.0  # 0 - 255 to 0.0 - 1.0
        print(path)
        if img_yolop_input.ndimension() == 3:
            img_yolop_input = img_yolop_input.unsqueeze(0)
        [pred, anchor_grid], seg, ll = segmentation_model(img_yolop_input)
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        da_seg_mask = driving_area_mask(seg)

        # Edge filtering and lane clustering
        ll_seg_mask = lane_line_mask(ll)

        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        show_seg_result(im0s, (da_seg_mask, ll_seg_mask), is_demo=True)

        output_path = os.path.join(output_folder, os.path.basename(path))
        gen_path = output_path+".png"
        gen_path = os.path.join(output_folder, os.path.basename(path) + f"_{counter}.png")
        cv2.imwrite(gen_path, pred_im_resized)

        # Save the generated image for each step
        step_output_path = os.path.join(output_folder, os.path.basename(path) + f"_step_{counter}.png")
        cv2.imwrite(step_output_path, pred_im_resized)


        if dataset.mode == 'image':
            cv2.imwrite(output_path, im0s)
            print(f" The image with the result is saved in: {output_path}")
        else:  # 'video' or 'stream'
            if vid_path != output_path:  # new video
                vid_path = output_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w,h = im0s.shape[1], im0s.shape[0]
                else:  # stream
                    fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    output_path += '.mp4'
                vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0s)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
