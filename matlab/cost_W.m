function c = cost_W(s,t)

  c = 0;
  for r=1:min(s,t)
    c = c + nchoosek(s+t,r)*nchoosek(s+t-r,r);
  end

end
