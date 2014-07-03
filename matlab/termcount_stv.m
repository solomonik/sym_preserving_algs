function c = termcount(s,t,v)
  w = s+t+v;

  orig_c = nchoosek(s+t,s)

  Z_c = nchoosek(w,s+v)*nchoosek(w,t+v)
  
  N_c = 0;
  for p=0:v
    for q=0:v
      for r=0:v-1
        if (p+q+r<=v & v-p-r<=t & v-q-r<=s)
          N_c = N_c + nchoosek(v,p)*nchoosek(v-p,q)*nchoosek(v-p-q,r)*nchoosek(s+t,s+v-p-r)*nchoosek(s+t,t+v-q-r);
        end
      end
    end
  end
  N_c

  M_c = 0;
  for r=1:min(s,t)
%     nchoosek(s+t,r)*nchoosek(s+t-r,s-r)*nchoosek(s+t-r,t-r)
    M_c = M_c + nchoosek(s+t,r)*nchoosek(s+t-r,s-r)*nchoosek(t,t-r);
  end
  M_c
  N_c+M_c

  c = orig_c - Z_c + M_c + N_c;

%  U_c = v^2*nchoosek(w-1,s+v)*nchoosek(w-1,t+v)
%  V_c = v*nchoosek(w-1,s+v)*nchoosek(w,t+v)+v*nchoosek(w,s+v)*nchoosek(w-1,t+v)
%  W_c = (s+t)*nchoosek(s+t-1,s)*nchoosek(s+t-1,t)
%  if (s+t>3)
%    X_c = nchoosek(s+t,s+t-2)*nchoosek(s+t-2,s)*nchoosek(s+t-2,t)
%  else
%    X_c = 0;
%  end
%  if (s+t>5)
%    Q_c = nchoosek(s+t,s+t-3)*nchoosek(s+t-3,s)*nchoosek(s+t-3,t)
%  else
%    Q_c = 0;
%  end
%
%  c = Z_c + U_c + X_c - V_c - W_c - Q_c - orig_c;

end
