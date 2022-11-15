%% plot color pics
clear; clc;
load(['./simulation_results/results/','truth','.mat']);

% load(['./simulation_results/results/','hdnet','.mat']);
% pred_block_hdnet = pred;

% load(['./simulation_results/results/','mst_s','.mat']);
% pred_block_mst_s = pred;
% 
% load(['./simulation_results/results/','mst_m','.mat']);
% pred_block_mst_m = pred;

% load(['./simulation_results/results/','mst_l','.mat']);
% pred_block_mst_l = pred;

% load(['./simulation_results/results/','mst_plus_plus','.mat']);
% pred_block_mst_plus_plus = pred;

load(['./simulation_results/results/','TwIST','.mat']);
pred_block_twist = pred;

load(['./simulation_results/results/','GAP-TV','.mat']);
pred_block_gaptv = pred;

load(['./simulation_results/results/','DeSCI','.mat']);
pred_block_desci = pred;

load(['./simulation_results/results/','dgsmp','.mat']);
pred_block_dgsmp = pred;

load(['./simulation_results/results/','hdnet','.mat']);
pred_block_hdnet = pred;

load(['./simulation_results/results/','mst_l','.mat']);
pred_block_mst_l = pred;

load(['./simulation_results/results/','cst_l_plus','.mat']);
pred_block_cst_l_plus = pred;

load(['./simulation_results/results/','dluf_mixs2','.mat']);
pred_block_dluf_mixs2 = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

truth(find(truth>0.7))=0.7;
% pred_block_hdnet(find(pred_block_hdnet>0.7))=0.7;
% pred_block_mst_s(find(pred_block_mst_s>0.7))=0.7;
% pred_block_mst_m(find(pred_block_mst_m>0.7))=0.7;
% pred_block_mst_l(find(pred_block_mst_l>0.7))=0.7;
% pred_block_mst_plus_plus(find(pred_block_mst_plus_plus>0.7))=0.7;

pred_block_twist(find(pred_block_twist>0.7))=0.7;
pred_block_gaptv(find(pred_block_gaptv>0.7))=0.7;
pred_block_desci(find(pred_block_desci>0.7))=0.7;
pred_block_dgsmp(find(pred_block_dgsmp>0.7))=0.7;
pred_block_hdnet(find(pred_block_hdnet>0.7))=0.7;
pred_block_mst_l(find(pred_block_mst_l>0.7))=0.7;
pred_block_cst_l_plus(find(pred_block_cst_l_plus>0.7))=0.7;
pred_block_dluf_mixs2(find(pred_block_dluf_mixs2>0.7))=0.7;


f = 5;

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(sum(squeeze(truth(f, :, :, :)), 3), [170 130 30 30]);
rect2crop=round(rect2crop)
% close(123);
imshow(yx / 28)
figure; 

% spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
% spec_mean_hdnet = mean(mean(squeeze(pred_block_hdnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
% spec_mean_mst_s = mean(mean(squeeze(pred_block_mst_s(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
% spec_mean_mst_m = mean(mean(squeeze(pred_block_mst_m(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
% spec_mean_mst_l = mean(mean(squeeze(pred_block_mst_l(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
% spec_mean_mst_plus_plus = mean(mean(squeeze(pred_block_mst_plus_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);

spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_twist = mean(mean(squeeze(pred_block_twist(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_gaptv = mean(mean(squeeze(pred_block_gaptv(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_desci = mean(mean(squeeze(pred_block_desci(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dgsmp = mean(mean(squeeze(pred_block_dgsmp(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_hdnet = mean(mean(squeeze(pred_block_hdnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_l = mean(mean(squeeze(pred_block_mst_l(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_cst_l_plus = mean(mean(squeeze(pred_block_cst_l_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dluf_mixs2 = mean(mean(squeeze(pred_block_dluf_mixs2(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);


% spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
% spec_mean_hdnet = spec_mean_hdnet./max(spec_mean_hdnet);
% spec_mean_mst_s = spec_mean_mst_s./max(spec_mean_mst_s);
% spec_mean_mst_m = spec_mean_mst_m./max(spec_mean_mst_m);
% spec_mean_mst_l = spec_mean_mst_l./max(spec_mean_mst_l);
% spec_mean_mst_plus_plus = spec_mean_mst_plus_plus./max(spec_mean_mst_plus_plus);

spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_twist = spec_mean_twist./max(spec_mean_twist);
spec_mean_gaptv = spec_mean_gaptv./max(spec_mean_gaptv);
spec_mean_desci = spec_mean_desci./max(spec_mean_desci);
spec_mean_dgsmp = spec_mean_dgsmp./max(spec_mean_dgsmp);
spec_mean_hdnet = spec_mean_hdnet./max(spec_mean_hdnet);
spec_mean_mst_l = spec_mean_mst_l./max(spec_mean_mst_l);
spec_mean_cst_l_plus = spec_mean_cst_l_plus./max(spec_mean_cst_l_plus);
spec_mean_dluf_mixs2 = spec_mean_dluf_mixs2./max(spec_mean_dluf_mixs2);


% corr_hdnet = roundn(corr(spec_mean_truth(:),spec_mean_hdnet(:)),-4);
% corr_mst_s = roundn(corr(spec_mean_truth(:),spec_mean_mst_s(:)),-4);
% corr_mst_m = roundn(corr(spec_mean_truth(:),spec_mean_mst_m(:)),-4);
% corr_mst_l = roundn(corr(spec_mean_truth(:),spec_mean_mst_l(:)),-4);
% corr_mst_plus_plus = roundn(corr(spec_mean_truth(:),spec_mean_mst_plus_plus(:)),-4);

corr_twist = roundn(corr(spec_mean_truth(:),spec_mean_twist(:)),-4);
corr_gaptv = roundn(corr(spec_mean_truth(:),spec_mean_gaptv(:)),-4);
corr_desci = roundn(corr(spec_mean_truth(:),spec_mean_desci(:)),-4);
corr_dgsmp = roundn(corr(spec_mean_truth(:),spec_mean_dgsmp(:)),-4);
corr_hdnet = roundn(corr(spec_mean_truth(:),spec_mean_hdnet(:)),-4);
corr_mst_l = roundn(corr(spec_mean_truth(:),spec_mean_mst_l(:)),-4);
corr_cst_l_plus = roundn(corr(spec_mean_truth(:),spec_mean_cst_l_plus(:)),-4);
corr_dluf_mixs2 = roundn(corr(spec_mean_truth(:),spec_mean_dluf_mixs2(:)),-4);


X = lam28;

Y(1,:) = spec_mean_truth(:); 
% Y(2,:) = spec_mean_hdnet(:); Corr(1)=corr_hdnet;
% Y(3,:) = spec_mean_mst_s(:); Corr(2)=corr_mst_s;
% Y(4,:) = spec_mean_mst_m(:); Corr(3)=corr_mst_m;
% Y(5,:) = spec_mean_mst_l(:); Corr(4)=corr_mst_l;
% Y(6,:) = spec_mean_mst_plus_plus(:); Corr(5)=corr_mst_plus_plus;
Y(2,:) = spec_mean_twist(:); Corr(1)=corr_twist;
Y(3,:) = spec_mean_gaptv(:); Corr(2)=corr_gaptv;
Y(4,:) = spec_mean_desci(:); Corr(3)=corr_desci;
Y(5,:) = spec_mean_dgsmp(:); Corr(4)=corr_dgsmp;
Y(6,:) = spec_mean_hdnet(:); Corr(5)=corr_hdnet;
Y(7,:) = spec_mean_mst_l(:); Corr(6)=corr_mst_l;
Y(8,:) = spec_mean_cst_l_plus(:); Corr(7)=corr_cst_l_plus;
Y(9,:) = spec_mean_dluf_mixs2(:); Corr(8)=corr_dluf_mixs2;


createfigure(X,Y,Corr)


