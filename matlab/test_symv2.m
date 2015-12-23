function test_symv2(ns)
  ntest = size(ns,2);

  rel_err_Ab = zeros(ntest,1);
  rel_err_pAb = zeros(ntest,1);
  for i=1:ntest
    if (ns(i) > 1)
      n = ns(i);
      %b = rand(n,1)-.5
%      b = [-1+3./n:2./n:1+1./n]'
      b = ones(n,1)-.9;
      for j=1:n/2
        b(2*j-1)=-.1;
      end
%      A=rand(n,n)-.5;
%      pA=rand(n,n);
%      for j=1:n
%        b(j) = (j-1)/(n-1)-.5;% = r(n,1)-.5;
%      end
%      A=ones(n,n)-.5;
      pA=zeros(n,n);
      for k=1:n
        for j=1:n
          pA(k,j)=mod(2*k+j-2,n)*.3+.001;
        end
      end

      c_fast = symv(pA,b);
      c_stnd = pA*b;
%      c_ans = pA*b;
%      pc_ans = pA*b;
%    
%      c = symv(A,b);
      c_ans = -.1*.3*n./2;
%      c_fast(1), c_fast(2), c_fast(3), c_ans
%      pc = symv(pA,b);
%      pc_ans = pA*b;

      %rel_err_fast(i) = norm(c_ans-c_fast)/norm(c_ans);
      %rel_err_stnd(i) = norm(c_ans-c_stnd)/norm(c_ans);
      rel_err_fast(i) = norm(c_fast-c_ans)/(norm(pA)*norm(b));
      rel_err_stnd(i) = norm(c_stnd-c_ans)/(norm(pA)*norm(b));
      %rel_err_Ab(i) = norm(c);%/norm(c_ans);
      %rel_err_pAb(i) = norm(pc);%/norm(pc_ans);
    end
  end
%  set(0,'defaultAxesFontName', 'Arial')
%set(0,'defaultTextFontName', 'Arial')
%  set(0,'defaultAxesFontSize',20);
%set(0,'defaultTextFontSize',20)
  [rel_err_fast; rel_err_stnd]
  set(0,'DefaultTextFontname', 'Times New Roman');
  loglog(ns,rel_err_fast,'-or',ns,rel_err_stnd,'-*b');
  hleg=legend('symmetry preserving algorithm relative error','direct evaluation algorithm relative error');
  set(hleg,'FontSize',14); %,'FontWeight','bold')
  set(gca,'GridAlpha',.5);
%  loglog(ns,rel_err_Ab,'-or');
%  legend('positive random A relative error')
  xlabel('dimension of A and b','FontSize',14); %,'FontWeight','bold');
  ylabel('Relative forward error with respect to exact solution','FontSize',14);%3,'FontWeight','bold');
  title('Relative error of c=A*b with positive A and alternating b','FontSize',14,'FontWeight','normal');%,'FontWeight','bold');
  ax = gca;
  ax.GridAlpha = .25;
  ax.MinorGridAlpha = .8;
  ax.MinorGridColor = [.15 .15 .15];
  grid on;
%  set(findall(gcf,'type','text')) 
%  set(gca,'FontSize',15,'fontWeight','bold')
%  set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
  set(findall(gcf,'type','axes'),'FontSize',14);%3,'FontWeight','bold')
%  set(findall(gcf,'type','text'),'fontSize',16) 
end
