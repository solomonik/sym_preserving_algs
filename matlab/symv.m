function c = symv(A,b)
  n = size(A,1);

  c = zeros(n,1);
  
%  for i=1:n
%    for k=n:-1:1
%      c(i)=c(i)+A(i,k)*b(k);
%    end
%  end

  z = zeros(n,1);
  sA = zeros(n,1);
  for i=1:n
    for k=1:n
      z(i) = z(i) + A(i,k)*(b(i)+b(k));
      sA(i) = sA(i)+A(i,k);
    end
    c(i) = z(i) - sA(i)*b(i);
  end


end
