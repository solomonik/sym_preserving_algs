function A = syr2k(x,y,sgn)
  if (nargin < 3) 
    sgn = 1;
  end
%  if (sgn)
%    A = x*y' + y*x';
%  else
%    A = x*y' - y*x';
%  end
  n = size(x,1);
  k = size(x,2);
  A=zeros(n,n);
  d=zeros(n);
  for i=1:n
    d(i) = x(i,:)*y(i,:)';
  end
  
  for i=1:n
    for j=1:i-1
      if (sgn==1)
        z = (x(i,:)+x(j,:))*(y(i,:)'+y(j,:)')-d(i)-d(j);;
        A(i,j) = z;
        A(j,i) = z;
      else 
        z = (x(i,:)-x(j,:))*(y(i,:)'+y(j,:)')+(-d(i)+d(j));
        A(i,j) = z;
        A(j,i) = -z;
      end    
    end
    if (sgn)
      A(i,i)=2*d(i);
    end
  end
end
