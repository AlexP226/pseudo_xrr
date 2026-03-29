function y = ave_vf( alpha_fc , footprint, energy, alpha_i, Ddet )
% averaged Vineyard factor along the footprint
% fun = @(l) T_length(alpha_fc, l);
% y = (integral(fun, -25, 25) )/50;
step = floor(footprint/5);
for i=1:step+1
    temp(i) = vf_length_corr(alpha_fc, -5*step/2+(i-1)*5, energy, alpha_i, Ddet );
end;
y = sum(temp(:)) / (step+1);

end

