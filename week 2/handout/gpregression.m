function [GPMean, GPVariance] = GPRegression(XTrain, yTrain, XTest, gamma, sigma, h)
  n = size(XTest,1);

  GPMean = zeros(n,1);
  GPVariance = zeros(n,n);
end

