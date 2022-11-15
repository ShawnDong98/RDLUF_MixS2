%% plot color pics
clear; clc;
load('./simulation_results/results/RDLUF_MixS2_9stage.mat');
save_file = './simulation_results/results/RDLUF_MixS2_9stage/rgb_results/';
% save_file = './simulation_results/truth/';
% load('./scene1_meas_H_truth.mat');
% save_file = './H_file/';

mkdir(save_file);
close all;
for i = 1:10
    recon = squeeze(pred(i,:,:,:));
    intensity = 5;
    for channel=1:28
        img_nb = [channel];  % channel number
        row_num = 1; col_num = 1;
        lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
            516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
            614.5 625.0 636.5 648.0];
        recon(find(recon>1))=1;
        name = [save_file 'frame' num2str(i) 'channel' num2str(channel)];
        dispCubeAshwin(recon(:,:,img_nb),intensity,lam28(img_nb), [] ,col_num,row_num,0,1,name);
    end
end
close all;


