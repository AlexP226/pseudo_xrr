function fresnel = GIXOS_fresnel(Qz, Qc )
% Fresnel reflectivity
r = (Qz-sqrt(Qz.^2-Qc^2))./(Qz+sqrt(Qz.^2-Qc^2));
refl = r.*conj(r);
refl(refl>1)=1;
fresnel = [Qz refl];

end

