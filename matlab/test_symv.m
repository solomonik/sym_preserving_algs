function test_symv(ns)
  ntest = size(ns,1);

  rel_err_Ab = zeros(ntest,1);
  rel_err_pAb = zeros(ntest,1);
  for i=1:ntest
    if (ns(i) > 1)
      n = ns(i);
      b = rand(n,1)-.5;
      A=rand(n,n)-.5;
      pA=rand(n,n);
%      for j=1:n
%        b(j) = (j-1)/(n-1)-.5;% = r(n,1)-.5;
%      end
%      A=ones(n,n)-.5;
%      pA=ones(n,n);

%      c_ans = pA*b;
%      pc_ans = pA*b;
%    
      c = symv(A,b);
      c_ans = A*b;
      pc = symv(pA,b);
      pc_ans = pA*b;

      rel_err_Ab(i) = norm(c_ans-c)/norm(c_ans);
      rel_err_pAb(i) = norm(pc_ans-pc)/norm(pc_ans);
      %rel_err_Ab(i) = norm(c);%/norm(c_ans);
      %rel_err_pAb(i) = norm(pc);%/norm(pc_ans);
    end
  end
%  set(0,'defaultAxesFontName', 'Arial')
%set(0,'defaultTextFontName', 'Arial')
%  set(0,'defaultAxesFontSize',20);
%set(0,'defaultTextFontSize',20)
  [rel_err_Ab, rel_err_pAb]
  loglog(ns,rel_err_pAb,'-or',ns,rel_err_Ab,'-*g');
  hleg=legend('positive random A relative error','random A relative error');
  set(hleg,'FontSize',13,'FontWeight','bold')
%  loglog(ns,rel_err_Ab,'-or');
%  legend('positive random A relative error')
  xlabel('dimension of A and b','FontSize',13,'FontWeight','bold');
  ylabel('Relative forward error with to respect standard algorithm','FontSize',13,'FontWeight','bold');
  title('Relative error of c=A*b using fast symmetric algorithm','FontSize',13,'FontWeight','bold');
%  set(findall(gcf,'type','text')) 
%  set(gca,'FontSize',15,'fontWeight','bold')
%  set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
  set(findall(gcf,'type','axes'),'FontSize',13,'FontWeight','bold')
%  set(findall(gcf,'type','text'),'fontSize',16) 
end
