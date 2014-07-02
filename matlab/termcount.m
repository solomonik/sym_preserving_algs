function t = termcount(s)

 nchoosek(3*s,s)^2  - 2*s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s)- 2*s*nchoosek(2*s-1,s)^2 + s*nchoosek(3*s-1,s-1)^2 
 nchoosek(2*s,s) 
%t = nchoosek(3*s,s)^2  - nchoosek(2*s,s) - 2*nchoosek(3*s-1,s-1)*nchoosek(3*s,s)- 2*s*nchoosek(2*s-1,s)^2 + s*nchoosek(3*s-1,s-1)^2 
  if (s==1)
    t = nchoosek(3*s,s)^2  - nchoosek(2*s,s) - s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) - 2*s*nchoosek(2*s-1,s)^2 - s*nchoosek(3*s-1,s-1)*nchoosek(2*s,s) %*nchoosek(s,s-1) %- nchoosek(2*s %+ s*nchoosek(3*s-1,s-1)^2 
  elseif (s==2)
   nchoosek(3*s,s)^2 - s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) - 2*s*nchoosek(2*s-1,s)^2 - s*nchoosek(3*s-1,s-1)*nchoosek(2*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(3*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(2*s,s)
    %t = nchoosek(3*s,s)^2  - nchoosek(2*s,s) - s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) - 2*s*nchoosek(2*s-1,s)^2 - s*nchoosek(3*s-1,s-1)*nchoosek(2*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(3*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(2*s,s)
    t = nchoosek(3*s,s)^2  - nchoosek(2*s,s) - s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) - 2*s*(nchoosek(2*s-1,s)^2) - s*nchoosek(3*s-1,s-1)*nchoosek(2*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(3*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(2*s,s)
  elseif (s==3)
    t = nchoosek(3*s,s)^2  - nchoosek(2*s,s) - s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) - 2*s*(nchoosek(2*s-1,s)^2) - s*nchoosek(3*s-1,s-1)*nchoosek(2*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(3*s,s) + nchoosek(s,2)*nchoosek(3*s-2,s-2)*nchoosek(2*s,s) + nchoosek(s,3)*nchoosek(3*s-3,s-3)*nchoosek(3*s,s) + nchoosek(s,3)*nchoosek(3*s-3,s-3)*nchoosek(2*s,s) 
  end

end
