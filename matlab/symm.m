function y = symm(A,x,sgn)
  if (nargin < 3) 
    sgn = 1;
  end
%  y = A*x;

  n = size(A,1);
  k = size(x,2);
  d = zeros(n);
  y = zeros(n,k);
  for i=1:n
    d(i) = sum(A(i,:));
  end
  
  for i=1:n
    for j=1:i-1
      if (sgn==1)
        z = A(i,j)*(x(i,:)+x(j,:));
        y(i,:) = y(i,:)+z;
        y(j,:) = y(j,:)+z;
      else 
        z = A(i,j)*(x(i,:)-x(j,:));
        y(i,:) = y(i,:)+z;
        y(j,:) = y(j,:)+z;
      end    
    end
    if (sgn==1)
      y(i,:)=y(i,:)+2*A(i,i)*x(i,:);
    end
  end
  for i=1:n
    y(i,:) = y(i,:)-d(i)*x(i,:);
  end
  
end
