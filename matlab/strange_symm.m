function C = strange_symm(A, B, sgn, complex)

  [mA,nA] = size(A);
  [mB,nB] = size(B);
  assert(mA == nA);
  assert(mB == nB);
  assert(mB == mA);
  n = nA;

  TA = A;
  TB = B;

  if (complex)
    C = zeros(n,n);

    for i=1:n
      C(i,i)=dot(A(i,1:n),B(1:n,i)');
    end
    for i=1:n
      C2(i)=dot(A(i,1:n),B(1:n,i));
    end
    for i=1:n
      C3(i)=dot(A(i,1:n)',B(1:n,i)');
    end

    if (sgn)
      for i=1:(n-1)
        for j=1:(n-i)
          for k=1:n
            TA(j+i,k) = A(j,k) + A(j+i,k)';
            TB(k,j+i) = B(k,j) + B(k,j+i)';
          end
        end
        for j=1:(n-i)
          C(j+i,j) = dot(TA(j+i,1:n),TB(1:n,j+i))-C2(j)-C3(j+i);
        end
      end
      C=C+C';
    else
      for i=1:(n-1)
        for j=1:(n-i)
          for k=1:n
            TA(j+i,k) = A(j,k) - A(j+i,k)';
            TB(k,j+i) = B(k,j) + B(k,j+i)';
          end
        end
        for j=1:(n-i)
          C(j+i,j) = dot(TA(j+i,1:n),TB(1:n,j+i))-C2(j)+C3(j+i);
        end
      end
      C=C'-C;
    end
  else
    C = zeros(n,n);

    for i=1:n
      C(i,i)=dot(A(i,1:n),B(1:n,i));
    end

    if (sgn)
      for i=1:(n-1)
        for j=1:(n-i)
          TA(j+i,1:n) = A(j,1:n) + A(j+i,1:n);
          TB(1:n,j+i) = B(1:n,j) + B(1:n,j+i);
        end
        for j=1:(n-i)
          C(j+i,j) = dot(TA(j+i,1:n),TB(1:n,j+i))-C(j,j)-C(j+i,j+i);
        end
      end
      C=C+C';
    else
      for i=1:(n-1)
        for j=1:(n-i)
          TA(j+i,1:n) = A(j,1:n) + A(j+i,1:n);
          TB(1:n,j+i) = B(1:n,j) - B(1:n,j+i);
        end
        for j=1:(n-i)
          C(j+i,j) = dot(TA(j+i,1:n),TB(1:n,j+i))-C(j,j)+C(j+i,j+i);
        end
      end
      C=C-C';
    end
  end
end
