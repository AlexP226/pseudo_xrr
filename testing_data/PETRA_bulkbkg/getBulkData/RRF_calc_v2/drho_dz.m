function drho_dz = drho_dz(sld)
%compute the gradient of the sld profile
%   sld array format: z[A], sld [E-6A-2], ascend z
drho_dz = sld;
delta_z = (sld(end,1)-sld(1,1))/(length(sld(:,1))-1);

drho_dz(:,3) = (sld(:,2) - [sld(2:end,2);sld(end,2)])./(sld(:,1) - [sld(2:end,1);sld(end,1)+delta_z]); %[E-6A^-2]

% %% plot
% close(findobj('name','drho_dz'));
% fig1=figure('name','drho_dz');
% yyaxis left;
% plot(drho_dz(:,1),drho_dz(:,2),'-k','LineWidth', 1.5);
% ylabel(['\rho [E-6' char(197) '^-^2]'],'FontSize',18);
% 
% hold on;
% yyaxis right;
% plot(drho_dz(:,1),drho_dz(:,3),'-.b', 'LineWidth', 1);
% ylabel('d\rho/dz','FontSize',18);
% hold off;
% 
% ax=gca;
% xlabel(['z [' char(197) ']'],'FontSize',18);
% ax.FontSize = 18;
% ax.LineWidth = 2;
% ax.TickDir = 'out';
% ax.YAxis(1).Color = 'k';
% ax.YAxis(2).Color = 'b';

end

