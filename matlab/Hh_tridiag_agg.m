function [B,U,V,D] = Hh_tridiag(A,b)

  n = size(A,1);
  
  B = A;

  U = zeros(n,n);
  V = zeros(n,n);

  V(1:n,1)=0;
  for j = 1:b:n-2,
    for i = j:min(n-2,j+b-1)
      
      % compute Householder vector
      U(1:i,i) = 0;
      U(i+1:n,i) = qr(B(i+1:n,i)- U(i+1:n,j:i-1) * V(i,j:i-1)' - V(i+1:n,j:i-1) * U(i,j:i-1)');
      %U(i+1:n,i) = qr(B(i+1:n,i)-syr2k( U(i+1:n,j:i-1), V(i,j:i-1)));
      
      % set u and tau
      %u = (i+1:n,i); 
      U(i+1,i) = 1;
      tau = 2/(U(:,i)'*U(:,i));
      
      % update trailing matrix
      %y = (B - U(:,j:i-1) * V(:,j:i-1)' - V(:,j:i-1) * U(:,j:i-1)')*U(:,i);
      y = symm(B - syr2k(U(:,j:i-1), V(:,j:i-1)),U(:,i));
      v = tau*y - tau^2/2 * (y'*U(:,i))*U(:,i);
      V(1:n,i)=v;
    end
%    B = B - U(:,j:j+b-1)*V(:,j:j+b-1)' - V(:,j:j+b-1)*U(:,j:j+b-1)';
    B = B -syr2k(U(:,j:min(n-2,j+b-1)),V(:,j:min(n-2,j+b-1)));
  end
  
  % check eigenvalues
  D = diag(diag(B))+diag(diag(B,-1),-1)+diag(diag(B,-1),1);
  norm(sort(eig(D))- sort(eig(A)),2)


end
