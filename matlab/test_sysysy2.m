function test_sysysy(ns)
  ntest = size(ns,2);

  rel_err_stnd = zeros(ntest,1);
  rel_err_fast = zeros(ntest,1);
  for i=1:ntest
    if (ns(i) > 1)
      n = ns(i);

      % build householder rank-1 Q = (I-2uuT) which is symmetric and compute  2I = Q*Q + Q*Q
      x = rand(n,1);
      nrmx = norm(x,2);
      u=x;
      u(1) = u(1) - nrmx;
      u = u./norm(u,2);
      A = (eye(n)-2.*u*u');

      C_stnd = A*A+A*A;
    
      C_fast = sysysy(A,A);

      C_ans = 2.*eye(n);

      rel_err_stnd(i) = norm(C_ans-C_stnd)/norm(C_ans);
      rel_err_fast(i) = norm(C_ans-C_fast)/norm(C_ans);
    end
  end
  rel_err_fast
  rel_err_stnd
  set(0,'DefaultTextFontname', 'Times New Roman');
  loglog(ns,rel_err_fast,'-or',ns,rel_err_stnd,'-*b');
  hleg=legend('symmetry preserving algorithm relative error','direct evaluation algorithm relative error');
  set(hleg,'FontSize',14,'FontWeight','normal')
  set(gca,'GridAlpha',.5);
  xlabel('dimension of A','FontSize',14,'FontWeight','normal');
  ylabel('Relative forward error with respect to exact solution','FontSize',14,'FontWeight','normal');
  title('Relative error of squaring a Householder transformation','FontSize',14,'FontWeight','normal');
  ax = gca;
  ax.GridAlpha = .25;
  ax.MinorGridAlpha = .8;
  ax.MinorGridColor = [.15 .15 .15];
  grid on;
  set(findall(gcf,'type','axes'),'fontSize',14,'FontWeight','normal')
end
