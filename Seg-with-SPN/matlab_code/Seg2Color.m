function res_img = Seg2Color(res,num_obj)
     ann = res;
     ann_img_1  = zeros(size(ann));
     ann_img_2  = 0.25*ones(size(ann));
     ann_img_2(res==0)       = 0;
     ann_img_3  = zeros(size(ann)); 
     for obj_id = 1:num_obj
        ann_img_1(res == obj_id)  = sin(pi*obj_id/num_obj/2)/2;
        ann_img_3(res == obj_id)  = cos(pi*obj_id/num_obj/2)/2; 
     end  
     res_img  = cat(3, ann_img_1,ann_img_2,ann_img_3);
     res_img  = uint8(255*res_img);

end