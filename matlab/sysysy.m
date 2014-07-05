function C = sysysy(A,B)
  n = size(A,1);

  Z = zeros(n,n);
  for i=1:n
    for j=1:n
      for k=1:n
        Z(i,j)=Z(i,j)+(A(i,j)+A(i,k)+A(j,k))*(B(i,j)+B(i,k)+B(j,k));
      end
    end
  end

  sA = zeros(n,1);
  sB = zeros(n,1);
 
  for i=1:n
    for k=1:n
      sA(i) = sA(i) + A(i,k);
      sB(i) = sB(i) + B(i,k);
    end
  end

  V = zeros(n,n);
  for i=1:n
    for j=1:n
      V(i,j) = n*A(i,j)*B(i,j)+A(i,j)*(sB(i)+sB(j))+(sA(i)+sA(j))*B(i,j);
    end
  end

  U = zeros(n,1);
  for i=1:n
    for k=1:n
      U(i) = U(i) + A(i,k)*B(i,k);
    end
  end
  
  W = zeros(n,n);
  for i=1:n
    for j=1:n
      W(i,j) = U(i)+U(j);
    end
  end

  C=zeros(n,n);
  for i=1:n
    for j=1:n
      C(i,j)=Z(i,j)-V(i,j)-W(i,j);
    end
  end
end
