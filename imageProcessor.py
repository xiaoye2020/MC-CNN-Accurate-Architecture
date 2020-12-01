import argparse
import cv2
import os

parser = argparse.ArgumentParser(description='image_process')
parser.add_argument('--left_data_path', default='F:/Dataset/KITTI 2015/data_scene_flow/training/image_2')
parser.add_argument('--right_data_path', default='F:/Dataset/KITTI 2015/data_scene_flow/training/image_3')
parser.add_argument('--dispar_data_path', default='F:/Dataset/KITTI 2015/data_scene_flow/training/disp_noc_0')
parser.add_argument('--save_path', default='date/')
parser.add_argument('--dataset_neg_low', default=4)
parser.add_argument('--dataset_neg_high', default=10)
parser.add_argument('--dataset_pos', default=1)
args = parser.parse_args()


def getFlist(path):
    for root, dirs, files in os.walk(path):
        pass

    return files

def main():
    f=open('processedData.txt', "a+")
  
    left_path = getFlist(args.left_data_path)
    right_path=getFlist(args.right_data_path)
    dispar_path=getFlist(args.dispar_data_path)

    for index in range(len(left_path)):
        left_img=cv2.imread(args.left_data_path+'/'+left_path[index])
        right_img=cv2.imread(args.right_data_path+'/'+right_path[index])
        dispar_img=cv2.imread(args.dispar_data_path+'/'+dispar_path[index],cv2.IMREAD_GRAYSCALE)

        for y in range(left_img.shape[0]):
            for x in range(left_img.shape[1]):
                dispar=dispar_img[y,x]
                if (dispar==0)|(y<4)|(y>left_img.shape[0]-5)|(x<4)|(x>left_img.shape[1]-5):
                    continue

                save_name=left_path[index].split(".")[0]+'_%d_%d'%(y,x)
                left_patch_path=args.save_path+save_name+"_left.png"
                save_left_image=left_img[y-4:y+5, x-4:x+5]
                cv2.imwrite(left_patch_path,save_left_image)
                # positive
                for pos in range(-args.dataset_pos,args.dataset_pos):
                    if (x-dispar+pos<4)|(x-dispar+pos>left_img.shape[1]-5):
                        continue
                    right_patch_path=args.save_path+save_name+'_%d.png'%(pos)
                    save_image=right_img[y-4:y+5,x-dispar+pos-4:x-dispar+pos+5]
                    cv2.imwrite(right_patch_path,save_image)
                    f.write(left_patch_path+' '+right_patch_path+' 1\n')
                
                for neg in range(-args.dataset_neg_high,-args.dataset_neg_low):
                    if (x-dispar+neg<4)|(x-dispar+neg>left_img.shape[1]-5):
                        continue
                    right_patch_path=args.save_path+save_name+'_%d.png'%(neg)
                    save_image=right_img[y-4:y+5,x-dispar+neg-4:x-dispar+neg+5]
                    cv2.imwrite(right_patch_path,save_image)
                    f.write(left_patch_path+' '+right_patch_path+' 0\n')
                
                for neg in range(args.dataset_neg_low,args.dataset_neg_high):
                    if (x-dispar+neg<4)|(x-dispar+neg>left_img.shape[1]-5):
                        continue
                    right_patch_path=args.save_path+save_name+'_%d.png'%(neg)
                    save_image=right_img[y-4:y+5,x-dispar+neg-4:x-dispar+neg+5]
                    cv2.imwrite(right_patch_path,save_image)
                    f.write(left_patch_path+' '+right_patch_path+' 0\n')
    f.close()
if __name__ == '__main__':
   main()