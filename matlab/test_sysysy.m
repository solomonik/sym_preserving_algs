function test_sysysy(ns)
  ntest = size(ns,1);

  rel_err_AA = zeros(ntest,1);
  rel_err_AB = zeros(ntest,1);
  for i=1:ntest
    if (ns(i) > 1)
      n = ns(i);
      A=rand(n,n)-.5;
      A=A+A';
      B=rand(n,n)-.5;
      B=B+B';

      C_ans = A*B+B*A;
    
      C = sysysy(A,B);

      C_ans2 = A*A+A*A;
    
      C2 = sysysy(A,A);


      rel_err_AB(i) = norm(C_ans-C)/norm(C_ans);
      rel_err_AA(i) = norm(C_ans2-C2)/norm(C_ans2);
    end
  end
  rel_err_AB
  rel_err_AA
  loglog(ns,rel_err_AA,'-or',ns,rel_err_AB,'-*g');
  hleg=legend('\Phi(A,A) relative error','\Phi(A,B) relative error')
  set(hleg,'FontSize',13,'FontWeight','bold')
  xlabel('dimension of A and B','FontSize',13,'FontWeight','bold');
  ylabel('Relative forward error with respect to standard algorithm','FontSize',13,'FontWeight','bold');
  title('Relative error of Jordan matrix multiplication using fast symmetric algorithm','FontSize',13,'FontWeight','bold');
  set(findall(gcf,'type','axes'),'fontSize',13,'FontWeight','bold')
end
