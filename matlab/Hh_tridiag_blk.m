function [B,U,V,D] = Hh_tridiag_blk(A,b)

  n = size(A,1);
  
  B = A;

  for i = 1:b:n-2,
    
    % compute Householder vector
    %U(1:i,i+1) = 0;
    [Q1,R] = qr(B(i+b:n,i:i+b-1))

    [U1,T,S]=q2y(Q1(:,1:b));
    for j=1:b
      B(i+j,i+j-1:i+b-1)=R(j,j:b);
      B(i+j-1:i+b-1,i+j)=R(j,j:b)'
      U1(j,j)=1;
      U1(j,j+1:b)=0;
    end
    U1=U1*diag(S);
    
%    norm((eye(n-b)-U1*T*U1')*Q1(:,1:b)-[eye(b);zeros(n-2*b,1)],2)
    
    % set u and tau
    %u = (i+1:n,i); 
    %U(i+1,i+1) = 1;
    %tau = 2/(U(:,i+1)'*U(:,i+1));
    
    % update trailing matrix
    %y = B * U(:,i+1);
    Y1 = B(i+b:n,i+b:n) * U1 * T;
    V1 = Y1 - .5.* (U1*T'*(U1'*Y1));
    B(i+b:n,i+b:n) = B(i+b:n,i+b:n) - U1*V1' - V1*U1';
    B(i+b+1:n,i:i+b-1)=0;
    B(i:i+b-1,i+b+1:n)=0;
    U(i:i+b-1,i:i+b-1)=0;
    U(i+b:n,i:i+b-1)=U1;
    V(i+b:n,i:i+b-1)=V1;
  end
  
  % check eigenvalues
  D = B;
%diag(diag(B))+diag(diag(B,-1),-1)+diag(diag(B,-1),1);
  [sort(eig(D)) sort(eig(A))]

end
