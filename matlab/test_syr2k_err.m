function test_syr2k_err(m,mat_szs,eps)
  n = size(mat_szs,1);

  if (nargin<3) 
    eps = 1E-9;
  end

  rel_err_syr2k = zeros(n,1);
  rel_err_fast_syr2k = zeros(n,1);
  rel_err_fast_syr2k_vz = zeros(n,1);

  mat_szs = round(mat_szs); 
 
  for i=1:n
    if (mat_szs(i) > 1)
      v = rand([m,mat_szs(i)])-.5;
      z = eps.*(rand([m,mat_szs(i)])-.5);
      R = rand([mat_szs(i),mat_szs(i)])-5.;
%      S = R + R';
      S = eye(mat_szs(i));
      w = v+z;
      corr_ans = v*z'-z*v';
      %norm(corr_ans)
      %norm(Z)
      Z= v*w'; %-w*v';
      syr2k_ans = Z-Z';
  %    fast_syr2k_ans = zeros(m,m);
  %    for j=1:10:mat_szs(i)
  %      fast_syr2k_ans = fast_syr2k_ans+...
  %        syr2k(v(:,j:min(j+9,mat_szs(i))), w(:,j:min(j+9,mat_szs(i))),0);
  %    end
      fast_syr2k_vz_ans = syr2k(v,z,0);
      fast_syr2k_ans = syr2k(v,w,0);
      rel_err_syr2k(i,1) = norm(syr2k_ans-corr_ans)/norm(corr_ans);
      rel_err_fast_syr2k(i,1) = norm(fast_syr2k_ans-corr_ans)/norm(corr_ans);
      rel_err_fast_syr2k_vz(i,1) = norm(fast_syr2k_vz_ans-corr_ans)/norm(corr_ans);
    end
  end
  [rel_err_syr2k, rel_err_fast_syr2k]
  loglog(mat_szs,rel_err_fast_syr2k,'-*g',mat_szs,rel_err_syr2k,'-or', mat_szs,rel_err_fast_syr2k_vz,'-xb');
  legend('\Phi(A,A*S+eps*B) error','\Psi(A,A*S+eps*B) error','\Phi(A,eps*B) error','Location','East');
  xlabel('# of columns in random matrices A and B');
  ylabel('Relative forward error with respect to \Phi(A,eps*B)');
  title('Relative error of C=A*(A*S+eps*B)^T-(A*S+eps*B)*A^T');
end
