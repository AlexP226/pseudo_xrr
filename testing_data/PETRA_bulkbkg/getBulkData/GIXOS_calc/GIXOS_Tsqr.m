function Tsqr = GIXOS_Tsqr(Qz, Qc, energy, alpha_i, Ddet, footprint )
% transmission with footprint average

planck = 1240.4;

Tsqr = Qz;
Tsqr(:,2) = asind(Tsqr(:,1)./(2*pi)*(planck/energy*10)-sind(alpha_i));  % alpha_f column
Tsqr(:,3) = Tsqr(:,2) ./ asind(Qc/(2*2*pi/(planck/energy*10)));         % x column
for i=1:size(Tsqr,1) 
    Tsqr(i,4)=ave_vf(Tsqr(i,2),footprint, energy, alpha_i, Ddet); 
end;

end

