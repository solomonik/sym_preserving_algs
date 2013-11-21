function [Q,R,Y,T] = hh_tsqr(A)

  n = size(A,1);
  m = size(A,2);

  Y = zeros(size(A,1), size(A,2));
  R = A;
  T = zeros(size(A,2), size(A,2));
  for i=1:m
    s = norm(R(i:n,i),2);
    if (R(i,i) > 0)
      alpha = -s;
    else
      alpha = s;
    end
    u      = R(i:n,i);
    u(1,1)   = u(1,1)-alpha;    %u(1,1) = A(i,i) +- norm
    nu     = norm(u,2);

    Y(i:n,i)   = u./nu;
    R(i,i)   = alpha;           %R(i,i) = +- norm
    R(i+1:n,i)   = 0;
    R(i:n,i+1:m) = (eye(n-i+1)-2.*Y(i:n,i)*Y(i:n,i)')*R(i:n,i+1:m);
    T(i,i)   = -2;
    T(1:i-1,i)   = -2.*T(1:i-1,1:i-1)*(Y(1:n,1:i-1)'*Y(1:n,i));
  end
  R = R(1:m,1:m);
  Q=(eye(n)+Y*T*Y');
end

