%% Qxy dependency display for thin film %%
% Chen 06.06.2023
% bkg subtraction
% bkg is extrapolated linearly

clear all;
close all;
%%
colorset = [    0 0.4470 0.7410; ...
                0.8500 0.3250 0.0980; ...
                0.9290 0.6940 0.1250; ...
                0.4940 0.1840 0.5560; ...
                0.4660 0.6740 0.1880; ...
                0.3010 0.7450 0.9330; ...
                0.6350 0.0780 0.1840; ...
                0 0.4470 0.7410; ...
                0.8500 0.3250 0.0980; ...
                0.9290 0.6940 0.1250; ...
    ];

%% geometry
gamma_E = double(eulergamma);
geo.Qc =0.0216;
geo.energy=15000; 
geo.alpha_i=0.07;
geo.wavelength = 12404/geo.energy;
geo.qxy0 = [0.01, 0.015, 0.02:0.01:0.12]; % a group of qxy0
geo.qxy_bkg = [0.35:0.01:0.45]; % bkg region from the GISAXS data itself
geo.DSresHW = 0.003; % HW of the DS resolution at specular
geo.DSqxyHW = 2*geo.DSresHW; % HW of the region of interest in qxy0 for integration, set to 2* DS resolution (res = 0.003A^-1)
%%%%%%%%%%%%%%%%%%%%%%
%% file
%%%%%%%%%%%%%%%%%%%%%%
% file
path_out = 'D:/github/pseudo_xrr/testing_data/PETRA_bulkbkg/';

% path = 'U:/p08/2023/data/11016139/processed/GID/angularrebin/';
% sample = 'pp1_edta_b_1';
% scan = 10;
% I0_sample_chamber = 5.1/5.0; % 5.1 for edta_a_1 #5, 5.0 for chamber bkg and ca_a_1 1141
% bkgsample = 'chamber_bkg';
% bkgscan = 1144;

% path = 'U:/p08/2023/data/11017009/processed/SC_ana_result/GID/anglerebin/';
% sample = 'h2o_20degc';
% scan = 263;
% I0_sample_chamber = 1; 
% bkgsample = 'trough_chamber_bkg';
% bkgscan = 319;

path = 'U:/p08/2023/data/11019137/processed/GID/angularrebin/';
sample = 'h2o_22degc';
scan = 1188;
I0_sample_chamber = 1; 
bkgsample = 'chamber_bkg';
bkgscan = 1190;

%%
% file name of the GISAXS to be imported
fileprefix = strcat('GID_',sample,'_',num2str(scan,'%05d'),'_angle');
Ifilename = strcat(path, fileprefix, '_I.dat');
tthfilename = strcat(path, fileprefix, '_tth.dat');
ttfilename = strcat(path, fileprefix, '_tt.dat');
importdata.Intensity = load(Ifilename);
importdata.tth = load(tthfilename);
importdata.tt = load(ttfilename);

% file name of the bkg GISAXS to be imported
bkgfileprefix = strcat('GID_',bkgsample,'_',num2str(bkgscan,'%05d'),'_angle');
bkgIfilename = strcat(path, bkgfileprefix, '_I.dat');
bkgtthfilename = strcat(path, bkgfileprefix, '_tth.dat');
bkgttfilename = strcat(path, bkgfileprefix, '_tt.dat');
importbkg.Intensityraw = load(bkgIfilename);
importbkg.Intensity = importbkg.Intensityraw*I0_sample_chamber;
importbkg.tth = load(bkgtthfilename);
importbkg.tt = load(bkgttfilename);

%% binning
binsize = 10;
groupnumber = floor(size(importdata.Intensity, 1)/binsize);

for groupidx = 1:groupnumber
    binneddata.Intensity(groupidx,:) = sum(importdata.Intensity((groupidx-1)*binsize+1:groupidx*binsize,:));
    binneddata.tt(groupidx,:) =  mean(importdata.tt((groupidx-1)*binsize+1:groupidx*binsize,:));
    binneddata.tth = importdata.tth;
    
    binnedbkg.Intensity(groupidx,:) = sum(importbkg.Intensity((groupidx-1)*binsize+1:groupidx*binsize,:));
    binnedbkg.tt(groupidx,:) =  mean(importbkg.tt((groupidx-1)*binsize+1:groupidx*binsize,:));
    binnedbkg.tth = importbkg.tth;
end

importdata = binneddata;
importbkg = binnedbkg;

%%
% angle step size:
geo.tth_step = mean(importdata.tth(2:end) - importdata.tth(1:end-1));
geo.tt_step = mean(importdata.tt(2:end) - importdata.tt(1:end-1));
% remove negative tt
tt_start_idx = find(importdata.tt<=0,1,'last');
importdata.Intensity(1:tt_start_idx,:) = [];
importdata.tt(1:tt_start_idx) = [];
importbkg.Intensity(1:tt_start_idx,:) = [];
importbkg.tt(1:tt_start_idx) = [];

%%
tth = asind(geo.qxy0 * geo.wavelength / 4 / pi)*2;   % tth for the qxy0
tth_bkg = asind(geo.qxy_bkg * geo.wavelength / 4 / pi)*2; % tth for the bkg region in GISAXS
for idx = 1:length(tth)
    tth_idx(idx) = find(importdata.tth>tth(idx),1,'first');
end
for idx = 1:length(tth_bkg)
    tth_bkg_idx(idx) = find(importdata.tth>tth_bkg(idx),1,'first');   % bkg tth in the GISAXS
end
tth_roiHW = rad2deg(geo.DSqxyHW .* geo.wavelength / 2 / pi ./ cosd(tth/2));   % region of interest of the tth
mat_roiHW = floor(tth_roiHW / geo.tth_step);
tth_roiHW_real = mat_roiHW*geo.tth_step;
geo.DSqxyHW_real = deg2rad(tth_roiHW_real)/2*4*pi/geo.wavelength .*cosd(tth/2);
%%
% arrange data structure
GIXOS.tt = importdata.tt;
geo.DSbetaHW = mean(GIXOS.tt(2:end) - GIXOS.tt(1:end-1))/2;
for idx = 1:length(tth)
    GIXOS.Qxy(:,idx) = 2*pi/geo.wavelength * sqrt((cosd(GIXOS.tt).*sind(tth(idx))).^2+(cosd(geo.alpha_i)-cosd(GIXOS.tt).*cosd(tth(idx))).^2);
end
GIXOS.Qz = 2*pi/geo.wavelength*(sind(GIXOS.tt)+sind(geo.alpha_i));
GIXOS.Q = sqrt(GIXOS.Qz.^2+GIXOS.Qxy.^2);
for idx = 1:length(tth)
    GIXOS.GIXOS_raw(:,idx) = sum(importdata.Intensity(:,tth_idx(idx)-mat_roiHW:tth_idx(idx)+mat_roiHW),2);
    GIXOS.GIXOS_bkg(:,idx) = sum(importbkg.Intensity(:,tth_idx(idx)-mat_roiHW:tth_idx(idx)+mat_roiHW),2);
end
for idx = 1:length(tth_bkg_idx)
    GIXOS.GIXOS_largetth_Qz((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1)=GIXOS.Qz(26:end);
    GIXOS.GIXOS_largetth_Qxy((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = 2*pi/geo.wavelength*sqrt((cosd(GIXOS.tt(26:end)).*sind(tth_bkg(idx))).^2+(cosd(GIXOS.tt(26:end)).*cosd(tth_bkg(idx))-cosd(geo.alpha_i)).^2);
    GIXOS.GIXOS_raw_largetth((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = sum(importdata.Intensity(26:end,tth_bkg_idx(idx)-mat_roiHW:tth_bkg_idx(idx)+mat_roiHW),2);
    GIXOS.GIXOS_bkg_largetth((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = sum(importbkg.Intensity(26:end,tth_bkg_idx(idx)-mat_roiHW:tth_bkg_idx(idx)+mat_roiHW),2);
end
GIXOS.GIXOS_largetth_Q = sqrt(GIXOS.GIXOS_largetth_Qz.^2+GIXOS.GIXOS_largetth_Qxy.^2);
%% bkg fit with Q
[GIXOS.GIXOS_largetth_Q ,sortIdx] = sort(GIXOS.GIXOS_largetth_Q ,'ascend');
GIXOS.GIXOS_largetth_Qz = GIXOS.GIXOS_largetth_Qz(sortIdx);
GIXOS.GIXOS_largetth_Qxy = GIXOS.GIXOS_largetth_Qxy(sortIdx);
GIXOS.GIXOS_raw_largetth = GIXOS.GIXOS_raw_largetth(sortIdx);
GIXOS.GIXOS_bkg_largetth = GIXOS.GIXOS_bkg_largetth(sortIdx);
GIXOS.bulk_bkg = GIXOS.GIXOS_raw_largetth - GIXOS.GIXOS_bkg_largetth;


bulkfittype = fittype("a+b*exp(x*c)",dependent="y", independent="x", coefficients = ["a" "b" "c"]);
f_baseline = fit(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg ,bulkfittype,'Upper',[mean(GIXOS.bulk_bkg(1:10),1)*5, 1000,10], 'Lower', [0, 0, 0], 'StartPoint', [mean(GIXOS.bulk_bkg(1:10),1), 100, 1]);
GIXOS.bulk_bkg_coeff = coeffvalues(f_baseline);
GIXOS.bulk_bkg_fit = exp(GIXOS.GIXOS_largetth_Q*GIXOS.bulk_bkg_coeff(3))*GIXOS.bulk_bkg_coeff(2)+GIXOS.bulk_bkg_coeff(1);

close(findobj('name','bulkbkg'));
op_bulkbkg = figure('name','bulkbkg','Position',[50,50,300,300]);
hold on
plot(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg, 'ko-');
plot(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg_fit,'r-');
hold off;

%%
bkgfilename = strcat(path_out,sample,'_',num2str(scan,'%05d'),'_bulkbkg.dat');
bkgfid = fopen(bkgfilename,'w');
fprintf(bkgfid,...
    '# path: %s\n# files\nsample file: %s\nbackground file: %s\nbulk bkg by fitting Q dependence for a sum of GIXOS at qxy_0 between %f and %f (parameters: %.3f + %.3f * exp(%.3f*Q))\n',...
    path, fileprefix, bkgfileprefix, geo.qxy_bkg(1), geo.qxy_bkg(end),GIXOS.bulk_bkg_coeff(1), GIXOS.bulk_bkg_coeff(2) ,GIXOS.bulk_bkg_coeff(3));
dlmwrite(bkgfilename,[GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg],'delimiter','\t','-append')
fclose(bkgfid);