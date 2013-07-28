perceptron <- function(data,labels,eta,nIter){
  # data is a design matrix of the form [x1,...,x_d]^(n), labels is (y1,...,yn)
  # eta is the learning rate
  # the perceptron finds a hyperplane that achieves linear separation between cases
  # if such a one exists
  # returns the weight vector learned after a specified number of iterations
  
  d = length(data[1,]) # dimensionality
  N = length(data[,1]) # number of training samples
  
  bias = rep(1,d)
  data = cbind(bias,data)
  d = length(data[1,]) # Adjust for bias
  
  weights = rep(0,d)
  
  # Perceptron functions
  
  f <- function(x,weights){
    y <- x %*% weights
    if(y >= 0){
      return(1)
    }
    else{
      return(-1)
    }
  }
  
  # Debug: First pass error count
  errcount <- 0
  for(n in 1:N){
    out <- f(data[n,],weights)
    if(out != labels[n]){
      errcount = errcount + 1
    }
  }
  cat("Errors before weight update: ", errcount, "\n")
  ############
  
  # Learning starts
  
  for(x in 1:nIter){
    for(n in 1:N){
      output <- f(data[n,],weights)
      if(output == labels[n]){
        weights <- weights
      }
      else{
        weights <- weights + eta * (data[n,] * labels[n])
      }
    }
    
  }
  
  # Debug: Errors after weights
  errcount <- 0
  for(n in 1:N){
    out <- f(data[n,],weights)
    if(out != labels[n]){
      errcount = errcount + 1
    }
  }
  cat("Errors after weight update: ", errcount, "\n")
  
  weights
}


# simulate training and test data

giveLabeledData <- function(N,dimen){
  dat = matrix(nrow=N,ncol=dimen,data=0)
  labs = rep(0,N)
  for(i in 1:N){
    # Draw sample from one of two distributions
    samp = rbinom(1,1,0.5)
    if(samp == 1){
      dat[i,] <- rnorm(dimen,mean = 10, sd = 0.5)
      labs[i] <- 1
    }
    else{
      dat[i,] <- rnorm(dimen,mean=1,sd=0.5)
      labs[i] <- -1
    }
  }
  return(list(data=dat,labels=labs))
}

mydata = giveLabeledData(130,2)

ab <- perceptron(mydata$data,mydata$labels,0.1,6)

# Plot decision boundary
# Obtained by solving for one of the input terms in y = w %*% x, 
# such that in the 2D case, x2 = w0/-w2 - w1/w2 * x1
# -w0/w2 is the intercept, -w1/w2 is the slope

plot(mydata$data,pch=20)
abline(-ab[1]/ab[3],ab[2]/-ab[3])
