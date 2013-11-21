function [U,V,D] = tridiagonalize(A)

  n = size(A,1);
  U = zeros(n,n);
  V = zeros(n,n);
  D = A;

  for i=1:1%n-1
    if (D(i+1,i) >= 0)
      alpha = -norm(D(i+1:n,i),2)
    else
      alpha = norm(D(i+1:n,i),2);
    end
    U(i+1,i) = sqrt(1+D(i+1,i)*D(i+1,i)/(alpha*alpha));
    U(i+2:n,i) = -D(i+2:n,i)./alpha

%    U(1:end,i)'*U(1:end,i)

    y = D*U(:,i);
    V(:,i) = y-.5.*(y'*U(:,i)).*U(:,i);
    D = D - U(:,i)*(V(:,i)') - V(:,i)*(U(:,i)');
  end
end
    


