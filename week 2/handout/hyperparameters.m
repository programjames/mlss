function [gamma, h,sigma] = HyperParameters (XTrain, yTrain,hs,sigmas)
  % Gamma is done for you, do a search for h,sigma
  gamma = 0.01*std(yTrain);
  h=1;
  sigma=1;
endfunction
