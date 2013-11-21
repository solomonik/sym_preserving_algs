function [B,U,V,D] = Hh_tridiag(A)

    n = size(A,1);
    
    B = A;

    U = zeros(n,n);
    V = zeros(n,n);

    V(1:n,1)=0;
    U(1:1,1)=1;
    for i = 1:n-2,
        
        % compute Householder vector
        U(1:i,i+1) = 0;
        U(i+1:n,i+1) = qr(B(i+1:n,i));
        
        % set u and tau
        %u = (i+1:n,i); 
        U(i+1,i+1) = 1;
        tau = 2/(U(:,i+1)'*U(:,i+1));
        
        % update trailing matrix
        y = B * U(:,i+1);
        v = tau*y - tau^2/2 * (y'*U(:,i+1))*U(:,i+1);
        B = B - U(:,i+1)*v' - v*U(:,i+1)';
        V(1:n,i+1)=v;
    end
    
    % check eigenvalues
    D = diag(diag(B))+diag(diag(B,-1),-1)+diag(diag(B,-1),1);
    [sort(eig(D)) sort(eig(A))]

end
