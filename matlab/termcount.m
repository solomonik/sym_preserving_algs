function t = termcount(s)

t = nchoosek(3*s,s)^2 - 2*s*nchoosek(3*s-1,s-1)*nchoosek(3*s,s) + nchoosek(2*s,2)*nchoosek(3*s-1,s-1)^2 - 2*s*nchoosek(2*s-1,s)^2 - nchoosek(2*s,s)

end
