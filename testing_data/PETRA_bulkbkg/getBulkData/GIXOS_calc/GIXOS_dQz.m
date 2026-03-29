function dQz = GIXOS_dQz(Qz, energy, alpha_i, Ddet, footprint )
% dQz from long footprint: +-footprint

planck = 1240.4;

dQz = Qz;
dQz(:,2) = asind(dQz(:,1)./(2*pi)*(planck/energy*10)-sind(alpha_i));  % alpha_f center
dQz(:,3) = atand(tand(dQz(:,2))*Ddet/(Ddet-footprint)); % alpha_f max
dQz(:,4) = atand(tand(dQz(:,2))*Ddet/(Ddet+footprint)); % alpha_f min

dQz(:,5) = ( (sind(dQz(:,3))+sind(alpha_i))*((2*pi)/(planck/energy*10)) - (sind(dQz(:,4))+sind(alpha_i))*((2*pi)/(planck/energy*10)) )*0.5;
dQz(:,6) = dQz(:,5)./dQz(:,1);
end

