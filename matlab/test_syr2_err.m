function test_syr2_err(ns,eps)
  n = size(ns,1);

  if (nargin<2) 
    eps = 1E-9;
  end

  rel_err_syr2 = zeros(n,1);
  rel_err_fast_syr2 = zeros(n,1);
  rel_err_fast_syr2_vz = zeros(n,1);
 
  for i=1:n
    if (ns(i) > 1)
      v = rand([ns(i),1])-.5;
      z = eps.*(rand([ns(i),1])-.5);
      R = rand([ns(i),ns(i)])-5.;
%      S = R + R';
      S = eye(ns(i));
      w = v+z;
      corr_ans = v*z'-z*v';
      %norm(corr_ans)
      %norm(Z)
      Z= v*w'; %-w*v';
      syr2_ans = Z-Z';
  %    fast_syr2_ans = zeros(m,m);
  %    for j=1:10:mat_szs(i)
  %      fast_syr2_ans = fast_syr2_ans+...
  %        syr2(v(:,j:min(j+9,mat_szs(i))), w(:,j:min(j+9,mat_szs(i))),0);
  %    end
      fast_syr2_vz_ans = syr2k(v,z,0);
      fast_syr2_ans = syr2k(v,w,0);
      rel_err_syr2(i,1) = norm(syr2_ans-corr_ans)/norm(corr_ans);
      rel_err_fast_syr2(i,1) = norm(fast_syr2_ans-corr_ans)/norm(corr_ans);
      rel_err_fast_syr2_vz(i,1) = norm(fast_syr2_vz_ans-corr_ans)/norm(corr_ans);
    end
  end
  [rel_err_syr2, rel_err_fast_syr2]
  loglog(ns,rel_err_fast_syr2,'-*g',ns,rel_err_syr2,'-or', ns,rel_err_fast_syr2_vz,'-xb');
  legend('\Phi(a,a*S+eps*b) error','\Psi(a,a*S+eps*b) error','\Phi(a,eps*b) error','Location','East');
  xlabel('# of columns in vectors a and b');
  ylabel('Relative forward error with respect to \Phi(a,eps*b)');
  title('Relative error of C=a*(a*S+eps*b)^T-(a*S+eps*b)*A^T');
end
