function [U,V,B] = reduce_band(A,b)
  n = size(A,1);
  
  if (nargin < 2) 
    b = 1;
  end

  if (n==1)
    B=A;
    U=1;
    V=1;
    return;
  end

  %U(1:b,1:b) = 0;
  B(1:b,1:b) = A(1:b,1:b);
  %[Q,B(b+1:b+b,1:b),U1,T] =  hh_tsqr(A(b+1:n,1:b));
  alpha = 0.0;
  for i=1:n-1
    alpha = alpha + A(i+1,1:b)*A(i+1,1:b);
  end
  alpha=sqrt(alpha);
  B(b+1,1)=A(b+1,1);
%  U1(1,1)=1;
  if (A(b+1,1) > 0)
    B1(b+1,1)=A(b+1,1)+alpha;
  else
    B1(b+1,1)=A(b+1,1)-alpha;
  end
  U1(1,1)=1;
  U1(2:n-1,1)=A(b+2:n,1)
  B(1:b,b+1:b+b)=B(b+1:b+b,1:b);
  %Y = symm(A(b+1:n,b+1:n),U(b+1:n,1:b))
  tau = 2/(U1'*U1);
  Y = tau.*A(b+1:n,b+1:n)*U1
  V1 = Y-.5.*tau.*(Y'*U1).*U1;
  

  %A(b+1:n,b+1:n) = A(b+1:n,b+1:n) - syr2k(U(b+1:n,1:b),V1);
  A(b+1:n,b+1:n) = A(b+1:n,b+1:n) - U1*V1'-V1*U1';

  V(b+1:n,1:b)=V1;  
  U(b+1:n,1:b)=U1;  

  if (n>b) 
    [U(b+1:n,b+1:n),V(b+1:n,b+1:n),B(b+1:n,b+1:n)] = reduce_band(A(b+1:n,b+1:n),b);
  end
    [sort(eig(B)) sort(eig(A))]
end
