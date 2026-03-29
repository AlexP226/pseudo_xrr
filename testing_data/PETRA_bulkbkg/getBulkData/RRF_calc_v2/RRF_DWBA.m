function RRF = RRF_DWBA(Qz,drho_dz)
%calculate R/Rf by distorted wave Born approximation
%version 2: any superphase
%   R/Rf = |structure_factor|^2 / (sld_inf-sld_0)^2
%   structure factor  = fourier(drho/dz)
%   low Qz correction by sqrt(Qz^2-Qc^2)
%% format Qz
if ~iscolumn(Qz)
    Qz = Qz';
end
RRF(1).Qz = Qz;
%% compute qc
RRF.sld_0 = mean(drho_dz(1:10,2));
RRF.sld_inf = mean(drho_dz(end-10:end,2));
RRF.Qc = 4*sqrt(pi*(RRF.sld_inf-RRF.sld_0)/10^6);
% for none-reflecting subphase
if abs(RRF.Qc)<1e-5
    %fprintf('subphase is matched with superphase\n');
    RRF.Qc = (1e-5)*1i;
    RRF.sld_inf = (RRF.Qc)^2/(16*pi)*10^6 + RRF.sld_0;
else
    %fprintf('subphase is optically different from the superphase\n');
end

%% Fourier transform
delta_z = (drho_dz(end,1)-drho_dz(1,1))/(length(drho_dz(:,1))-1);
Qz_mod = sqrt((RRF.Qz).^2-(RRF.Qc)^2);
RRF.Fstruct = zeros(length(RRF.Qz),1);
for k=1:length(Qz)
    RRF.Fstruct(k,1) = (drho_dz(:,3))'*exp(1i*Qz_mod(k)*drho_dz(:,1)*cos(pi))*delta_z;
end
phase_angle = atan(imag(RRF.Fstruct)./real(RRF.Fstruct));
RRF.RRF = (RRF.Fstruct).*conj(RRF.Fstruct)/(RRF.sld_inf-RRF.sld_0)^2;
RRF.RRF(RRF.Qz<RRF.Qc) = 1;

% %% plot RRF
% close(findobj('name','RRF_sim'));
% fig1=figure('name','RRF_sim');
% yyaxis left;
% plot(RRF.Qz, RRF.RRF, '-r','LineWidth',1.5);
% ylabel('|F|^2/\Delta\rho^2','FontSize',18);
% hold on;
% yyaxis right;
% plot(RRF.Qz, phase_angle, '-.b', 'LineWidth', 1);
% ylabel('\theta [rad]','FontSize',18);
% %plot(Qz, imag(Fstruct), '-.g', 'LineWidth', 1);
% hold off;
% legend('R/RF', 'phase');
% 
% ax=gca;
% xlabel(['Q_z [' char(197) '^-^1]'],'FontSize',18);
% ax.FontSize = 18;
% ax.LineWidth = 2;
% ax.TickDir = 'out';
% ax.YAxis(1).Color = 'k';
% ax.YAxis(2).Color = 'k';
% 
% assignin('base','Fstruct', RRF.Fstruct);

end

