################################################
#
#       Text Mining for Economics & Finance
#
#   Useful functions for handling word vectors
#
#
################################################


vecCheck<-function(text, vecdata, wfdata=NULL, 
                   PCAtrim=0, a=10^-3){
  
  if(class(vecdata)[1]!="data.table"){
    stop("Load vector data as data.table")
  }
  text<-gsub('(?<=[,:-;!.?])  (?=[,:-;!.?])',' ',gsub("([,:-;!?]|[.]+)", " \\1 ", text),perl=T)
  text<-doc2concrete:::ctxpand(text)
  
  .allCaps<-which((stringr::str_count(text,"[A-Z]")/nchar(text))>.4)
  text[.allCaps]<-tolower(text[.allCaps])
  
  dtm1<-quanteda::dfm(quanteda::tokens(text),tolower=F)
  dtm1<-dtm1[,colnames(dtm1)%in%vecdata$word]
  dtm1<-dtm1[,order(colnames(dtm1))]
  
  if(is.null(wfdata)){
    dtm1<-dtm1[,(colnames(dtm1)%in%vecdata$word)]
  } else{
    dtm1<-dtm1[,(colnames(dtm1)%in%vecdata$word)&(colnames(dtm1)%in%wfdata$Word)]
    weights<-(a/(a+wfdata$FREQavg[(wfdata$Word%in%colnames(dtm1))]))
  }
  vecd1<-vecdata[vecdata$word%in%colnames(dtm1),]
  vecs1<-vecd1[order(tolower(unlist(vecd1[,1]))),-1]
  
  scores<-matrix(NA,nrow(dtm1),ncol(vecs1))
  colnames(scores)<-colnames(vecs1)
  if(nrow(dtm1)>1000){
    tpb=txtProgressBar(0,nrow(dtm1))
  }
  for(x in 1:nrow(dtm1)){
    counts<-as.vector(dtm1[x,])
    if(sum(counts)==0){
      scores[x,]<-rep(0,ncol(vecs1))
    } else{
      idx=which(counts>0)
      ctx=counts[idx]
      vecrows<-unlist(lapply(1:length(idx), function(z) rep(idx[z],ctx[z])))
      if(is.null(wfdata)){
        scores[x,]<-colSums(vecs1[vecrows,])
      } else {
        scores[x,]<-apply(vecs1[vecrows,],2, weighted.mean, w=weights[vecrows])
      }
    }
    if(nrow(dtm1)>1000){
      setTxtProgressBar(tpb,x)
    }
  }
  
  eRows<-is.na(scores[,1])
  scores[is.na(scores)]<-0
  # Remove first (or more) Principal Components
  if(PCAtrim>0) {
    svd = svd(scores, nu = PCAtrim, nv = 0)
    PCs = as.data.frame(svd$u)
    names(PCs) = paste("PC", c(1:PCAtrim))
    fit = lm(scores~., data = PCs)
    scores = residuals(fit)
  }
  scores[eRows,]<-0
  return(scores)
}


vecSimCalc<-function(x=NULL,xvec=NULL,
                     y,
                     vecfile, wffile=NULL,
                     PCAtrim=0){
  
  if(length(y)>1){
    stop("One ground truth at a time!")
  }
  if(is.null(x)&is.null(xvec)){
    stop("Must include text or vectorized text as X")
  }
  if(is.null(x)){
    yvec=vecCheck(y,vecfile, wffile)
  } else {
    xyvec<-vecCheck(c(x,y),
                    vecfile, 
                    wffile,
                    PCAtrim=PCAtrim)
    if(length(x)==1){
      xvec<-matrix(xyvec[1:length(x),],nrow = 1)
    } else{
      xvec<-xyvec[1:length(x),]
    }
    yvec<-xyvec[nrow(xyvec),]
  } 
  
  mags=apply(xvec,1,function(z) sqrt(sum(z^2)))*sqrt(sum(yvec^2))
  dots=t(apply(xvec,1, function(z) sum(z*yvec)))
  sims=as.vector(dots/mags)
  return(sims)
}


bowSimCalc<-function(x,y){
  if(length(y)>1){
    stop("One ground truth at a time!")
  }
  counts<-quanteda::as.dfm(doc2concrete::ngramTokens(c(x,y)))
  xvec=counts[-nrow(counts),]
  yvec=counts[nrow(counts),]
  mags=apply(xvec,1,function(z) sqrt(sum(z^2)))*sqrt(sum(yvec^2))
  dots=t(apply(xvec,1, function(z) sum(z*yvec)))
  sims=as.vector(dots/mags)
  return(sims)
}

################################################
# Semantic Axis!
################################################

semaxis<-function(pairs,docs,vecdata){
  if(!(is.data.frame(pairs) & ncol(pairs)==2)){
    stop("Must input two-column dataframe")
  }
  ###STORE EMBEDDING AS MATRIX, NORMALIZE WORD VECTORS###
  cdfm<-as.matrix(column_to_rownames(vecdata,var = "word"))
  cdfmn<-t(apply(cdfm,1,nrm))
  
  # generate semantic axis from pairs
  word_dims<-data.frame(matrix(NA,nrow(pairs),300))
  for (j in 1:nrow(pairs)){
    rp_word1<-pairs[j,1]
    rp_word2<-pairs[j,2]
    tryCatch(word_dims[j,]<-nrm(nrm(cdfmn[rp_word1,])
                                -nrm(cdfmn[rp_word2,])),
             error=function(e){})
  }
  
  
  # save the axis
  dim_ave<-nrm(colMeans(word_dims, na.rm = TRUE))
  
  # encode docs as vectors
  docVec=vecCheck(docs,vecdata = vecdata)
  
  # calculate similarity of doc vectors to semantic axis
  projs<-apply(docVec,1,function(x) cos(x,dim_ave))
  
  return(projs)
}

##### helpful functions ##########

#Normalize vector#
nrm <- function(x) x/sqrt(sum(x^2))

#Cosine Similarity#
cos <- function(x,y) sum(x*y)/(sqrt(sum(x^2))*sqrt(sum(y^2)))
################################################
1

