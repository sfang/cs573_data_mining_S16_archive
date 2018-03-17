population1p = read.csv("population1p.csv",header=TRUE)
#gop = read.csv("GOP_donations_2012.csv",header=TRUE)

population = as.matrix(population1p$donates)
#gopdonations = as.matrix(gop$donation[gop$donation > 0])

N = round(nrow(population)/200)

# randomly split DEMs into N groups of average length nrow(demdonations)/N
demsplit = as.list(split(population, sample(1:N, nrow(population), replace=T)))

# find average of each bin
population_bin_averages = unlist(lapply(demsplit,mean))

# find 20 quantiles of each the batch averages
quantiles_dem = quantile(population_bin_averages,probs = seq(0, 1, 1/20))

# plot histogram
hist(population_bin_averages,main="Histogram of Population1p Batches (200 Samples)",freq=FALSE,breaks=quantiles_dem,xlim=c(0,10))


#N = round(nrow(gopdonations)/400)

# randomly split GOP into N groups of average length nrow(demdonations)/N
#gopsplit = split(gopdonations, sample(1:N, nrow(gopdonations), replace=T))

# find average of each bin
#gop_bin_averages = unlist(lapply(gopsplit,mean))

# find 20 quantiles of each the batch averages
#quantiles_gop = quantile(gop_bin_averages,probs = seq(0, 1, 1/20))

# plot histogram
#hist(gop_bin_averages,main="Histogram of GOP Batches (Average Donations, 400 Samples)",freq=FALSE,breaks=quantiles_gop,xlim=c(0,4000))

