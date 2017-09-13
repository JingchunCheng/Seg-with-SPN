clear all; close all; clc;
% This script shows a simple version of our CRAF process
% Download DAVIS dataset into folder 'data', and try this script

davis_dir   = '../data/DAVIS/';
img_dir     = [davis_dir,'JPEGImages/480p/'];
ann_dir     = [davis_dir,'Annotations/2017/'];

fgbg_dir     = '../results/fgbg/';
res_SPN_dir  = 'res_SPN/';
write_dir    = 'res_CRAF/';

video = 'running';
mkdir(write_dir);
mkdir([write_dir,video]);

ann     = imread([ann_dir,video,'/00000.png']);
num_obj = double(max(max(ann)));
ann_img  = Seg2Color(ann,num_obj);

images = dir(fullfile([img_dir,video,'/*.jpg']));
  for k = 1:20
      img = imread([img_dir,video,'/',images(k).name]);
      im_name = images(k).name;
      im_name = im_name(1:end-4);     
          
      if k == 1
          res = imread(sprintf('%s/%s/00000.png',ann_dir,video));
          imwrite(uint8(res),[write_dir,video,'/',im_name,'.png']);
          ref_img = Seg2Color(res,num_obj);
          imwrite(ref_img,[write_dir,video,'/',im_name,'.jpg']);
          imwrite(img + ref_img,[write_dir,video,'/',im_name,'_show.jpg']);
          continue
      end
      
 
     img_fgbg = imread([fgbg_dir,video,'/',im_name,'.png']);
      

     thre_area = 0.2;
     res_prev = imread(sprintf('%s/%s/%05d.png',write_dir,video,str2num(im_name)-1 ));
     for obj_id = 1:num_obj
        res_name = [res_SPN_dir,video,'/',im_name,'_',num2str(obj_id),'.png']; 
        res = imread(res_name);
        res_concat_ori(:,:,obj_id) = res;
  
        img_reg = regionprops(res>30,  'all');  
 
        areas = [img_reg.Area];
        J_objs = zeros(numel(areas),1);
        for j = 1:numel(areas) 
            res_prev_obj = (res_prev == obj_id);
            pixel_idx = img_reg(j).PixelIdxList;
            tmp = zeros(size(res));
            tmp(pixel_idx) = 1;
            [J_obj, inters, fp, fn] = jaccard_single(res_prev_obj, tmp);
            J_objs(j) = J_obj; 
        end   
        
        j = 1;
        while 1
            if length(J_objs) == 0
                break
            end
            if j > length(J_objs)
               j = 1;
            end
            
           area = img_reg(j).Area;
           if max(J_objs) <= 0.01
               res = res*0;
               break
           end 
            if J_objs(j) == max(J_objs) 
              if (area/max(areas) > thre_area)
                for nn = 1:numel(areas)
                    if nn == j
                        continue
                    else
                       pixel_idx = img_reg(nn).PixelIdxList; 
                       if res(pixel_idx(1)) ~= 255
                          res(pixel_idx) = 0;
                       end
                    end
                end
                break;
              else
                  pixel_idx = img_reg(j).PixelIdxList; 
                  res(pixel_idx) = 255;
                  J_objs(j) = 0;
              end  
            end
            j = j + 1;
        end
        
      res_concat(:,:,obj_id) = res;
     end 


     thre_fg = 100;
     
     [out,pos] = max(res_concat,[],3);
     pos(out<thre_fg) = 0;
     pos = pos.*(img_fgbg>0);     

     [out,pos_ori] = max(res_concat_ori,[],3);
     pos_ori(out<128) = 0;
     res_ori  = pos_ori.*(img_fgbg>0);
     res_img  = Seg2Color(res_ori,num_obj);
 
     J_thre = 0.2;
     for obj_id = 1:num_obj
         res_prev_obj = (res_prev == obj_id);
         if sum(sum(res_prev_obj)) < 100
             continue
         end
         if sum(sum(pos==obj_id)) < 0.3*sum(sum(res_prev_obj))
             for nn = 1:num_obj
               res = res_concat_ori(:,:,nn);  
               res(pos>0) = 0;
               img_reg = regionprops(res>0,  'all');  
               areas = [img_reg.Area]; 
               for mm = 1:numel(areas)
                 pixel_idx = img_reg(mm).PixelIdxList;
                 tmp = zeros(size(res));
                 tmp(pixel_idx) = 1;
                 [J_obj, inters, fp, fn] = jaccard_single(res_prev_obj, tmp);
                 disp(J_obj)
                 if J_obj > J_thre
                    pos(pixel_idx) = obj_id;
                 end         
               end  
             end
            J_objs(j) = J_obj; 
         end
     end

     
     pos = pos.*(img_fgbg>0);
     
     clear res_concat_ori 
     
     ref_img  = Seg2Color(pos, num_obj);
     
     subplot(3,2,1),imshow(img),title(sprintf('%s frm %05d',video,k-1));
     subplot(3,2,2),imshow(ann_img),title(sprintf('frm0 annt: num-obj = %d',num_obj));
     subplot(3,2,3),imshow(res_img),title('before process');
     subplot(3,2,4),imshow(img + res_img),title('before process');
     subplot(3,2,5),imshow(ref_img),title('after process');
     subplot(3,2,6),imshow(img + ref_img),title('after process');
     pause;
     
     imwrite(uint8(pos),[write_dir,video,'/',im_name,'.png']);
     imwrite(ref_img,[write_dir,video,'/',im_name,'.jpg']);
     imwrite( img + ref_img,[write_dir,video,'/',im_name,'_show.jpg']);
     
     clear res_concat
  end




      