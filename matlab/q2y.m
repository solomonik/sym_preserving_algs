function [Y,T,S,gf_lu] = q2y(Q)
	[n,b] = size(Q);
	Y = Q;
	S = ones(b,1);

  gf_lu=0;

	%% compute Y1 and W given Q
	for i = 1:b
		% Householder QR with hints (compute QR of A1 using R), or
		% LU without pivoting (compute LU of A1-S*R)
    
		% flip sign
 		if (sign(Y(i,i)) > 0) && (i <= n)
			S(i) = -1;
			Y(:,i) = -Y(:,i);
		end
    
    % compute row of W
    Y(i,i) = Y(i,i) - 1;

    if i < b
      % compute column of Y1
      Y(i+1:b,i) = Y(i+1:b,i) / Y(i,i);

      % update trailing matrix
      Y(i+1:b,i+1:b) = Y(i+1:b,i+1:b) - Y(i+1:b,i) * Y(i,i+1:b); 
    end

    if(nargout>3)
      gf_lu = max(gf_lu,max(max(abs(Y)))); 
    end


	end

	U = triu(Y(1:b,1:b)); % = -W = -T * Y1' 
	Y1 = tril(Y(1:b,1:b), -1) + eye(b);
	if nargout > 1
		T = -U / Y1';
	end

	%% compute Y2 given Q2 and W
	if n > b
    if(nargout>3)
      INVU = inv(U);
      Y2 = Y; 
      for i = 1:b
        Y2(b+1:end,i) = Y(b+1:end,1:b) * INVU(:,i); 
        gf_lu = max(gf_lu,max(max(abs(Y2))));
      end
    end

      Y(b+1:end,1:b) = Y(b+1:end,1:b) / U;
	end

  maxel = max(eps,max(max(abs(Q- eye(n,b) ))));
  gf_lu = gf_lu / maxel;
end


