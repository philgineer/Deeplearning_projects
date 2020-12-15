function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X*theta;

thetaExcludingZero = [ 0 ; theta(2:length(theta)) ];

reg = lambda/(2*m)*sum(thetaExcludingZero.^2);

J = 1/(2*m)*sum((hypothesis - y).^2) + reg;


%h = X * theta;
%squaredErrors = (h - y) .^ 2;
%thetaExcludingZero = [ [ 0 ]; theta([2:length(theta)])];
%J = (1 / (2 * m)) * sum(squaredErrors) + (lambda / (2 * m)) * sum(thetaExcludingZero .^ 2);


grad = 1/m*sum((hypothesis - y).*X) + lambda/m*thetaExcludingZero';

%왜 thetaExcludingZero'에서 transpose(')가 필요하지???

% =========================================================================

grad = grad(:);

end
