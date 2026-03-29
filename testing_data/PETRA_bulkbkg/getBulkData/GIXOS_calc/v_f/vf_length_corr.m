function y = vf_length_corr( alpha_fc, length, energy, alpha_i, Ddet )
% corrected vf along the footprint 
%   alpha_fc: center alpha_f on the footprint
%   length: displacement of the scattering point on the footprint
%   Ddet [mm]: distance of detector to rotation center

alpha_f = atand(Ddet*tand(alpha_fc)/(Ddet-length));

y = vineyard_factor(alpha_f, energy, alpha_i);


end

