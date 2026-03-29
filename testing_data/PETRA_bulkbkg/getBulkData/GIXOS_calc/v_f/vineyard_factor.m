function vf = vineyard_factor( alpha_f, energy, alpha_i )
% vineyard_factor
% = Transmission^2*penetration_depth
% penetration_depth takes accounts the paths -in and -out both.
% see Dosch's paper Phys Rev B (1987)
%   for now: energy  = 15keV

%%%%%%%%%%%%%%%%
% constant
%%%%%%%%%%%%%%%%
planck = 1240.4;
% for water
qc = 0.0216;
beta = 1*10^(-9);
alpha_c = asin(qc/(2*2*pi/(planck/energy*10)));

l_i = 1/sqrt(2)*sqrt(alpha_c^2 - (alpha_i/180*pi)^2+sqrt((alpha_c^2 - (alpha_i/180*pi)^2).^2+ (2*beta)^2));

%%%%%%%%%%%%%%%%
% calculation
%%%%%%%%%%%%%%%%

x = alpha_f/(alpha_c/pi*180);
if x>0
    % T=(abs(2*x./(x+sqrt(x.^2-1)+0.00085*sqrt(x.^2-1)*1i))).^2;
    T = (abs(2*x./(x+sqrt(x.^2-1-2*beta*1i/alpha_c^2)))).^2;
    l_f = 1/sqrt(2)*sqrt(alpha_c^2 - (alpha_f/180*pi).^2+sqrt((alpha_c^2 - (alpha_f/180*pi).^2).^2+ (2*beta)^2));
    %T=(abs(2*x./(x+sqrt(x.^2-1)))).^2;
else
    T=0;
    l_f = 1/sqrt(2)*sqrt(alpha_c^2 - (alpha_f/180*pi).^2+sqrt((alpha_c^2 - (alpha_f/180*pi).^2).^2+ (2*beta)^2));
%else
    %T=(2*x).^2;
end;

normalization = (planck/energy*10)/(2*pi)/l_i;   % normalization factor for vf: l_f at infinite alpha_f

vf = (planck/energy*10)/(2*pi)*T./(l_f+l_i)/normalization;
% see Dosch's paper Phys Rev B (1987)

end

