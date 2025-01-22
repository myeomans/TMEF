
kendall_acc<-function(x,y,percentage=TRUE){
  kt=cor(x,y,method="kendall")
  kt.acc=.5+kt/2
  kt.se=sqrt((kt.acc*(1-kt.acc))/length(x))
  report=data.frame(acc=kt.acc,
                    lower=kt.acc-1.96*kt.se,
                    upper=kt.acc+1.96*kt.se)
  report = round(report,4)
  if(percentage) report = report*100
  return(report)
}
