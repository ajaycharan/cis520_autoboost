function [H] = multi_entropy(p)
% MULTI_ENTROPY - Compute H(P(X)) for discrete multi-valued X.
%
% Usage:
% 
%    H = multi_entropy(P)
%
%  Returns the entropy H = -\sum_x p(x) * log(p(x)).
%  For an K X N matrix P, H is a 1 x N vector of entropy for each of the 
%  N distributions over K values.

    H = zeros(1, size(p,2));

    % iterate over N
    for i=1:size(p,2)

        % take K values
        p_x = p(:,i);

        % perform entropy calculation
        p_times_logp = min(0, p_x .* log2(p_x));
        H(1, i) = -sum(p_times_logp);
    end
end
